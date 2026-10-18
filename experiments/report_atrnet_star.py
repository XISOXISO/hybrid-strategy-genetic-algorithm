from __future__ import annotations
from pathlib import Path as _Path
import sys as _sys
_experiment_dir = _Path(__file__).resolve().parent
for _search_path in (_experiment_dir.parent, _experiment_dir):
    if str(_search_path) not in _sys.path:
        _sys.path.insert(0, str(_search_path))
import argparse, itertools, math, statistics
from collections import defaultdict
from pathlib import Path
from atrnet_star_common import atomic_csv, atomic_json, decode_candidate, digest, read_json, seed_for
from src.HGA.atrnet_star.journal import read_events
ALL_NAMES = ['RandomSearch', 'TPE', 'CMA_ES', 'GridSearchWithBudget', 'ASHA', 'EvolutionaryHPO', 'MomentumBasedRandomSearch', 'AdamHPO', 'GWO', 'PSO', 'GA_Basic', 'JSO', 'LSHADE', 'HGA_RM', 'HGA_CM', 'HGA_SW50', 'HGA_SW60']

def collect(outputs):
    reference = None
    tracks = {}
    source = {}
    for output in map(Path, outputs):
        p = read_json(output / 'protocol.json')
        identity = p['identity']
        b = identity['benchmark']
        if digest(identity) != p['protocol_id'] or digest(b) != identity['benchmark_id']:
            raise ValueError('Protocol checksum mismatch')
        if reference is not None and b != reference:
            raise ValueError('Cannot combine different frozen benchmark/evaluation protocols')
        reference = b
        for name in identity['algorithms']:
            for run in p['selected_runs']:
                key = (name, run)
                if key in tracks:
                    raise ValueError(f'Duplicate algorithm/run shard: {key}')
                jp = output / 'runs' / name / f'run_{run:02d}' / 'journal.sqlite'
                if not jp.exists():
                    raise ValueError(f'Missing journal: {jp}')
                events, meta = read_events(jp, return_meta=True)
                if meta.get('fingerprint') != digest([p['protocol_id'], name, run]):
                    raise ValueError('Journal task fingerprint mismatch')
                if int(meta.get('committed_fe', -1)) != sum((e['new_fe'] for e in events)):
                    raise ValueError('Journal progress metadata mismatch')
                for rec in events:
                    result = rec['result']
                    cid = rec['candidate_id']
                    if result['config'] != decode_candidate(cid) or result['training_seed'] != seed_for('candidate', b['seed'], run, cid):
                        raise ValueError('Configuration/training seed replay mismatch')
                    expected = 1 - result['f1'] + 1e-06 * min(result['loss'], 10.0)
                    if abs(result['objective'] - expected) > 1e-12:
                        raise ValueError('Objective/metrics mismatch')
                rows = [e for e in events if e['new_fe']]
                if len(rows) > b['max_fe']:
                    raise ValueError('Budget overrun')
                tracks[key] = rows
                source[key] = str(output)
    if reference is None:
        raise ValueError('No outputs')
    return (reference, tracks, source)

def selected(rows, limit=None):
    candidates = [r for r in rows if (limit is None or r['fe'] <= limit) and r['result']['status'] == 'complete']
    return max(candidates, key=lambda r: (r['result']['f1'], -r['result']['loss'])) if candidates else None

def report(outputs, destination, target=None, plots=False):
    b, tracks, sources = collect(outputs)
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    target = b['max_fe'] if target is None else target
    names = sorted({n for n, r in tracks})
    runsets = {n: {r for name, r in tracks if name == n} for n in names}
    paired = sorted(set.intersection(*runsets.values())) if runsets else []
    common = min((len(rows) for rows in tracks.values()), default=0)
    progress = []
    bestconfigs = []
    history = []
    for (name, run), rows in sorted(tracks.items()):
        best = selected(rows)
        progress.append({'algorithm': name, 'run': run, 'completed_fe': len(rows), 'target_fe': target, 'reached_target': len(rows) >= target, 'best_val_macro_f1': best['result']['f1'] if best else None, 'fe_at_best': best['fe'] if best else None, 'pruned': sum((r['result']['status'] == 'pruned' for r in rows)), 'numerical_failures': sum((r['result']['status'] == 'failed' for r in rows)), 'cache_hits': sum((r['result']['cache_hit'] for r in rows)), 'nominal_training_seconds': sum((r['result']['training_seconds'] for r in rows)), 'physical_training_seconds': sum((r['result']['physical_training_seconds'] for r in rows))})
        recovery = read_json(Path(sources[name, run]) / 'runs' / name / f'run_{run:02d}' / 'weight_reconstruction.json') if (Path(sources[name, run]) / 'runs' / name / f'run_{run:02d}' / 'weight_reconstruction.json').exists() else {'count': 0, 'seconds': 0.0}
        progress[-1].update(weight_reconstruction_count=recovery['count'], weight_reconstruction_seconds=recovery['seconds'], total_physical_training_seconds=progress[-1]['physical_training_seconds'] + recovery['seconds'])
        if best:
            import json
            bestconfigs.append({'algorithm': name, 'run': run, 'completed_fe': len(rows), 'selected_fe': best['fe'], 'candidate_id': best['candidate_id'], 'validation_macro_f1': best['result']['f1'], 'validation_loss': best['result']['loss'], 'best_epoch': best['result']['best_epoch'], 'training_seed': best['result']['training_seed'], 'config': json.dumps(best['result']['config'], sort_keys=True), 'source_output': sources[name, run]})
        best = None
        for r in rows:
            result = r['result']
            if result['status'] == 'complete' and (best is None or (result['f1'], -result['loss']) > (best['result']['f1'], -best['result']['loss'])):
                best = r
            history.append({'algorithm': name, 'run': run, 'fe': r['fe'], 'candidate_id': r['candidate_id'], 'candidate_f1': result['f1'], 'status': result['status'], 'best_f1': best['result']['f1'] if best else None, 'repaired': r['repaired'], 'cache_hit': result['cache_hit'], 'epochs_completed': result['epochs_completed'], 'training_seconds': result['training_seconds'], 'physical_training_seconds': result['physical_training_seconds']})
    checkpoints = sorted((c for c in {1, 5, 10, 20, 50, 100, 200, 250, 300, 400, 500, target, common} | set(range(50, b['max_fe'] + 1, 50)) if 0 < c <= common))
    summaries = []
    ranks = []
    values = {}
    from scipy.stats import rankdata
    for cp in checkpoints:
        vals = {name: [selected(tracks[name, r], cp)['result']['f1'] if selected(tracks[name, r], cp) else None for r in paired] for name in names}
        for name, v in vals.items():
            finite = [x for x in v if x is not None]
            summaries.append({'algorithm': name, 'checkpoint_fe': cp, 'paired_runs': len(paired), 'valid_runs': len(finite), 'mean': statistics.mean(finite) if finite else None, 'sd': statistics.stdev(finite) if len(finite) > 1 else None, 'min': min(finite) if finite else None, 'max': max(finite) if finite else None})
        if paired and all((all((x is not None for x in v)) for v in vals.values())):
            per_run = [rankdata([-vals[n][i] for n in names], method='average') for i in range(len(paired))]
            for i, name in enumerate(names):
                ranks.append({'algorithm': name, 'checkpoint_fe': cp, 'paired_runs': len(paired), 'mean_rank': statistics.mean((float(v[i]) for v in per_run))})
        if cp == common:
            values = vals
    full_runs = set(range(b['runs']))
    formal = b['mode'] == 'atr' and b['max_fe'] == 500 and (b['runs'] == 10) and all((runsets[n] == full_runs for n in names)) and all((len(v) >= 500 for v in tracks.values()))
    summary = {'benchmark_id': digest(b), 'mode': b['mode'], 'algorithms': names, 'paired_run_ids': paired, 'common_fe': common, 'completed_unique_fe': sum((len(v) for v in tracks.values())), 'target_unique_fe': target * len(tracks), 'all_reached_target': all((len(v) >= target for v in tracks.values())), 'formal_10_runs_complete_for_present_algorithms': formal, 'full_17_algorithm_formal_complete': formal and set(names) == set(ALL_NAMES), 'missing_global_run_ids': {n: sorted(full_runs - runsets[n]) for n in names}, 'test_evaluated': False, 'note': 'Scores are validation only. Unique FE is not equal GPU compute: ASHA prunes and cached identical full trials reuse results.'}
    stats = {'performed': False, 'reason': 'Requires all 10 formal runs at FE500 with valid complete-trial scores'}
    if formal and len(names) >= 3 and values and all((all((v is not None for v in a)) for a in values.values())):
        from scipy.stats import friedmanchisquare, wilcoxon
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            f = friedmanchisquare(*(values[n] for n in names))
        pairs = []
        for a, c in itertools.combinations(names, 2):
            x = values[a]
            y = values[c]
            p = 1.0 if all((i == j for i, j in zip(x, y))) else float(wilcoxon(x, y, alternative='two-sided').pvalue)
            pairs.append({'algorithm_a': a, 'algorithm_b': c, 'raw_p': p})
        order = sorted(range(len(pairs)), key=lambda i: pairs[i]['raw_p'])
        previous = 0.0
        for rank, index in enumerate(order):
            previous = max(previous, min(1.0, (len(pairs) - rank) * pairs[index]['raw_p']))
            pairs[index]['holm_p'] = previous
        atomic_csv(destination / 'wilcoxon_holm.csv', pairs)
        stats = {'performed': True, 'checkpoint_fe': 500, 'paired_runs': 10, 'friedman_statistic': float(f.statistic) if math.isfinite(f.statistic) else None, 'friedman_p': float(f.pvalue) if math.isfinite(f.pvalue) else None, 'holm_family_size': len(pairs)}
    from atrnet_star_final_evaluation import report_tests
    summary.update(report_tests(outputs, destination, tracks, b, formal))
    summary['full_17_algorithm_formal_complete'] = summary['full_17_algorithm_formal_complete'] and summary['test_complete_for_present_tracks']
    atomic_json(destination / 'statistics.json', stats)
    atomic_json(destination / 'progress.json', summary)
    atomic_csv(destination / 'task_progress.csv', progress)
    atomic_csv(destination / 'selected_configs.csv', bestconfigs, ['algorithm', 'run', 'completed_fe', 'selected_fe', 'candidate_id', 'validation_macro_f1', 'validation_loss', 'best_epoch', 'training_seed', 'config', 'source_output'])
    atomic_csv(destination / 'search_history.csv', history, ['algorithm', 'run', 'fe', 'candidate_id', 'candidate_f1', 'status', 'best_f1', 'repaired', 'cache_hit', 'epochs_completed', 'training_seconds', 'physical_training_seconds'])
    atomic_csv(destination / 'checkpoint_summary.csv', summaries, ['algorithm', 'checkpoint_fe', 'paired_runs', 'valid_runs', 'mean', 'sd', 'min', 'max'])
    atomic_csv(destination / 'average_rank.csv', ranks, ['algorithm', 'checkpoint_fe', 'paired_runs', 'mean_rank'])
    if plots and common and paired:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(11, 6))
        for name in names:
            curve = []
            for cp in range(1, common + 1):
                v = [selected(tracks[name, r], cp) for r in paired]
                curve.append(statistics.mean((x['result']['f1'] for x in v)) if all((x is not None for x in v)) else float('nan'))
            ax.plot(range(1, common + 1), curve, label=name, alpha=0.85)
        ax.set(xlabel='Unique FE', ylabel='Best validation macro-F1', title='ATRNet-STAR SOC10 N4800: common-budget comparison')
        ax.set_ylim(0, 1.01)
        ax.grid(alpha=0.25)
        ax.legend(fontsize=7, ncol=2)
        fig.tight_layout()
        fig.savefig(destination / 'convergence.png', dpi=180)
        fig.savefig(destination / 'convergence.pdf')
        plt.close(fig)
    return summary

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--outputs', nargs='+', required=True)
    p.add_argument('--report', required=True)
    p.add_argument('--target', type=int, default=500)
    p.add_argument('--require-full', action='store_true')
    p.add_argument('--require-all-algorithms', action='store_true')
    a = p.parse_args()
    result = report(a.outputs, a.report, a.target, plots=True)
    if a.require_full and (not result['formal_pipeline_complete_for_present_algorithms']):
        raise ValueError('Missing runs/FE; inspect progress.json')
    if a.require_all_algorithms and set(result['algorithms']) != set(ALL_NAMES):
        raise ValueError('Missing algorithms; inspect progress.json')
    print(result)
if __name__ == '__main__':
    main()
