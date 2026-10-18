import hashlib
import itertools
import math
import json
import random
import sqlite3
import time
from pathlib import Path
import numpy as np

GRID = np.array(list(itertools.product(range(3), repeat=8)), dtype=np.int8)

class Paused(BaseException):
    pass

class ReplayMismatch(BaseException):
    pass

class EvaluationFailure(BaseException):

    pass

def hash_json(x):
    return hashlib.sha256(json.dumps(x, sort_keys=True, ensure_ascii=False, allow_nan=False).encode()).hexdigest()

def atomic_json(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix+'.tmp')
    with tmp.open('w', encoding='utf-8') as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.flush()
        import os
        os.fsync(f.fileno())
    tmp.replace(path)

def config_id(genes):
    return int(np.dot(np.asarray(genes, dtype=int), 3**np.arange(7, -1, -1)))

class Journal:
    def __init__(self, path, backend, target, run, seed, fingerprint):
        self.path = Path(path); self.path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(self.path, timeout=120)
        self.db.execute('PRAGMA journal_mode=DELETE')
        self.db.execute('PRAGMA synchronous=FULL')
        self.db.execute('CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT)')
        self.db.execute('CREATE TABLE IF NOT EXISTS calls (seq INTEGER PRIMARY KEY, payload TEXT NOT NULL, checksum TEXT NOT NULL)')
        old = self.db.execute("SELECT value FROM meta WHERE key='fingerprint'").fetchone()
        if old and old[0] != fingerprint:
            raise RuntimeError('Protocol/code/environment changed: use a NEW output folder')
        self.db.execute("INSERT OR IGNORE INTO meta VALUES ('fingerprint', ?)", (fingerprint,))
        self.db.execute("INSERT OR IGNORE INTO meta VALUES ('committed_fe','0')")
        self.db.commit()
        self.saved = read_events(self.path)
        if self.db.execute('PRAGMA integrity_check').fetchone()[0] != 'ok':
            raise RuntimeError('SQLite integrity check failed')
        self.backend = backend; self.target = target; self.run = run; self.seed = seed
        self.seq = 0; self.fes = 0; self.seen = {}; self.best = None
        self.last_pruned = False; self.fingerprint = fingerprint
        self.new_trainings = 0

    def resolve(self, raw, repair):
        genes = np.rint(np.clip(raw, 0, 2)).astype(int)
        cid = config_id(genes)
        if not repair or cid not in self.seen:
            return genes, cid, False
        ids = np.array([i for i in range(6561) if i not in self.seen], dtype=int)
        if not len(ids):
            raise RuntimeError('Discrete search space exhausted')
        distances = ((GRID[ids].astype(float)-raw)**2).sum(axis=1)
        nearest = ids[np.isclose(distances, distances.min(), rtol=0, atol=1e-12)]

        tie = int(hash_json([self.seed, self.run, self.fes+1, raw.tolist()])[:16], 16)
        cid = int(nearest[tie % len(nearest)])
        return GRID[cid].copy(), cid, True

    def prefetch(self, proposals):
        if not hasattr(self.backend, 'prefetch_many'):
            return

        shadow = object.__new__(Journal)
        shadow.seen = dict(self.seen); shadow.fes = self.fes
        shadow.seed = self.seed; shadow.run = self.run
        plans = []
        for offset, vector in enumerate(proposals):
            if shadow.fes >= self.target:
                break
            raw = np.clip(np.asarray(vector, dtype=float), 0, 2)
            genes, cid, changed = shadow.resolve(raw, True)
            seq = self.seq + offset
            if seq < len(self.saved):
                rec = self.saved[seq]
                if (not np.allclose(raw, rec['proposal'], rtol=0, atol=1e-12)
                    or cid != rec['candidate_id'] or genes.tolist() != rec['genes']
                    or changed != rec['repaired']):
                    raise ReplayMismatch('Prefetch proposal mismatch')
            else:
                plans.append(dict(cid=cid, fe=shadow.fes+1))
            shadow.seen[cid] = {}; shadow.fes += 1
        self.backend.prefetch_many(plans)

    def __call__(self, vector, trial=None, repair=True, sampler_proposal=None):
        try:
            return self._evaluate(vector, trial=trial, repair=repair, sampler_proposal=sampler_proposal)
        except Exception as exc:
            raise EvaluationFailure(f'Evaluation/journal failed at call {self.seq+1}: {exc}') from exc

    def _evaluate(self, vector, trial=None, repair=True, sampler_proposal=None):
        if self.fes >= self.target:
            raise Paused()
        if self.seq > 2000*self.target:
            raise RuntimeError('Proposal guard reached; optimizer is stalled')
        raw = np.asarray(vector, dtype=float).reshape(-1)
        if raw.shape != (8,) or not np.isfinite(raw).all():
            raise RuntimeError('Invalid optimizer proposal')
        raw = np.clip(raw, 0, 2)
        genes, cid, repaired = self.resolve(raw, repair)
        cached_local = cid in self.seen
        if self.seq < len(self.saved):
            rec = self.saved[self.seq]
            if not np.allclose(raw, rec['proposal'], rtol=0, atol=1e-12) or cid != rec['candidate_id'] or rec['genes'] != genes.tolist() or rec['repaired'] != bool(repaired or sampler_proposal is not None) or rec.get('sampler_proposal') != sampler_proposal:
                raise ReplayMismatch(f'Proposal mismatch at call {self.seq+1}; do not continue or silently reset')
            result = rec['result']
            if trial is not None:
                for ep, val, expected_prune in result['reports']:
                    trial.report(val, ep)
                    if bool(trial.should_prune()) != expected_prune:
                        raise ReplayMismatch('ASHA replay pruning mismatch')
        else:
            if cached_local:
                result = dict(self.seen[cid]['result'])
            else:

                if getattr(self.backend, 'isolated_worker_rng', False):
                    result = self.backend.evaluate(genes.tolist(), cid, trial)
                else:
                    np_state, py_state = np.random.get_state(), random.getstate()
                    try:
                        result = self.backend.evaluate(genes.tolist(), cid, trial)
                    finally:
                        np.random.set_state(np_state); random.setstate(py_state)
                self.new_trainings += int(not result.get('cache_hit', False))
            rec = dict(proposal=raw.tolist(), genes=genes.tolist(), candidate_id=cid,
                       repaired=bool(repaired or sampler_proposal is not None), sampler_proposal=sampler_proposal, new_fe=not cached_local,
                       fe=self.fes + int(not cached_local), result=result)
            self.db.execute('INSERT INTO calls VALUES (?, ?, ?)',
                            (self.seq, json.dumps(rec, allow_nan=False), hash_json(rec)))
            self.db.execute("UPDATE meta SET value=? WHERE key='committed_fe'",(str(rec['fe']),))
            self.db.commit()
        if bool(rec['new_fe']) != (not cached_local):
            raise ReplayMismatch('Unique FE replay mismatch')
        self.seq += 1
        if not cached_local:
            self.fes += 1; self.seen[cid] = rec
            if result['status'] == 'complete':
                key = (result['f1'], -result['loss'])
                if self.best is None or key > (self.best['result']['f1'], -self.best['result']['loss']):
                    self.best = rec
        self.last_pruned = result['status'] == 'pruned'
        if self.seq > len(self.saved) and not cached_local:
            score = self.best['result']['f1'] if self.best else float('nan')
            if getattr(self.backend, 'isolated_worker_rng', False):
                with (self.path.parent/'runtime.log').open('a') as log:
                    log.write(f'FE {self.fes}/{self.target} best_val_F1={score:.6f}\n')
            else:print(f'FE {self.fes}/{self.target} best_val_F1={score:.6f}', flush=True)
            if hasattr(self.backend, 'committed'): self.backend.committed(self.fes, self.best)
        return float(result['objective'])

    def close(self):
        self.db.close()

def read_events(path, return_meta=False):
    if not Path(path).exists(): return ([], {}) if return_meta else []
    db = sqlite3.connect(Path(path).resolve().as_uri()+"?mode=ro", uri=True, timeout=30)
    try:
        db.execute("BEGIN")
        meta=dict(db.execute("SELECT key,value FROM meta")) if return_meta else {}
        records=[];seen=set();fe=0
        for seq,payload,checksum in db.execute('SELECT seq,payload,checksum FROM calls ORDER BY seq'):
            rec=json.loads(payload)
            if seq!=len(records) or hash_json(rec)!=checksum:
                raise ReplayMismatch('Journal sequence/checksum mismatch')
            cid=int(rec['candidate_id'])
            if config_id(rec['genes'])!=cid or any(g not in [0,1,2] for g in rec['genes']):
                raise ReplayMismatch('Journal genes/CID mismatch')
            fresh=cid not in seen
            if fresh!=rec['new_fe']:raise ReplayMismatch('Stored unique FE mismatch')
            fe+=int(fresh)
            if rec['fe']!=fe:raise ReplayMismatch('Stored FE sequence mismatch')
            result=rec['result']
            if result['status'] not in ['complete','pruned','failed'] or not math.isfinite(result['objective']) or not 0<=result['f1']<=1 or not math.isfinite(result['loss']) or result['loss']<0:
                raise ReplayMismatch('Invalid stored objective/metrics')
            seen.add(cid);records.append(rec)
        return (records,meta) if return_meta else records
    finally: db.close()
