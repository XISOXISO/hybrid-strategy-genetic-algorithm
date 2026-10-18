import copy, io, json, tempfile
from contextlib import redirect_stdout
from pathlib import Path
import numpy as np
if __package__:
    from .common import atomic_json,digest,read_json,runtime_hashes
    from .hga_optimizer import ControlledHGA,VARIANTS,SWITCH
    from .journal import Journal,Paused,read_events,hash_json,config_id
    from .shared_resources import SharedWeights,weight_identity
else:
    from common import atomic_json,digest,read_json,runtime_hashes
    from hga_optimizer import ControlledHGA,VARIANTS,SWITCH
    from journal import Journal,Paused,read_events,hash_json,config_id
    from shared_resources import SharedWeights,weight_identity

FIELDS=['generation','stagnation_counter','maintenance_events','switch_fe_actual','switch_generation',
        'history','best_score','best_genes','population','objectives','initial_diversity']

def export_state(c):
    state={k:getattr(c,k).tolist() if isinstance(getattr(c,k),np.ndarray) else copy.deepcopy(getattr(c,k)) for k in FIELDS}
    state['rng']=copy.deepcopy(c.rng.bit_generator.state);return state

def restore_state(c,state):
    if set(state)!=set(FIELDS+['rng']):raise ValueError('HGA state fields differ')
    for k in FIELDS:setattr(c,k,np.asarray(state[k],dtype=float) if k in ['population','objectives','best_genes'] else copy.deepcopy(state[k]))
    c.rng.bit_generator.state=copy.deepcopy(state['rng'])

def semantic(rec):
    rec=copy.deepcopy(rec)
    for k in ['cache_hit','physical_training_seconds','training_seconds','peak_vram_mb']:rec['result'].pop(k,None)
    return rec

def compatible(state,variant,budget,n=5):
    previous=n
    for entry in state['history']:
        fraction=SWITCH[variant]
        expected=('reverse' if variant=='RM' else 'conventional') if fraction is None else ('reverse' if previous/budget<fraction else 'conventional')
        if entry['assignment']!=expected:return False
        previous=entry['fe']
    return True

def verify(obj,args):
    clean=dict(obj);checksum=clean.pop('checksum',None)
    if checksum!=digest(clean):raise ValueError('Prefix checksum differs')
    expected=dict(protocol_id=args.protocol_id,run=args.run,seed=args.seed,budget=args.max_fe,engine_sha256=runtime_hashes()['hga_optimizer.py'])
    if obj['identity']!=expected:raise ValueError('Prefix source/run/seed/budget differs')

    with tempfile.TemporaryDirectory() as tmp:
        j=Journal(Path(tmp)/'verify.sqlite',None,obj['fe'],args.run,args.seed,'prefix-check')
        try:
            for i,rec in enumerate(obj['events']):j.db.execute('INSERT INTO calls VALUES (?,?,?)',(i,json.dumps(rec),hash_json(rec)))
            j.db.execute("UPDATE meta SET value=? WHERE key='committed_fe'",(str(obj['fe']),));j.db.commit();j.saved=read_events(j.path)
            c=ControlledHGA(obj['source_variant'],j,args.seed+args.run,args.max_fe)
            captured=[]
            def stop(controller):
                if j.fes==obj['fe']:captured.append(export_state(controller));raise Paused()
            c.boundary_hook=stop
            with redirect_stdout(io.StringIO()):
                try:c.run()
                except Paused:pass
            if len(captured)!=1 or captured[0]!=obj['state'] or j.seq!=len(obj['events']):raise ValueError('Prefix optimizer state/replay differs')
        finally:j.close()

def apply(c,j,args):
    folder=Path(args.output)/'shared_prefix'/f'run_{args.run:02d}'
    choices=[]
    for path in folder.glob(f'*_to_{c.variant}.json'):
        obj=read_json(path)
        if obj['fe']<=j.target:choices.append((obj['fe'],path,obj))
    for _,path,obj in sorted(choices,key=lambda item:(-item[0],str(item[1]))):
        verify(obj,args)
        if not compatible(obj['state'],c.variant,args.max_fe):raise ValueError('Prefix mutation assignments differ')
        common=min(len(j.saved),len(obj['events']))
        if any(semantic(a)!=semantic(b) for a,b in zip(j.saved[:common],obj['events'][:common])):

            raise ValueError('Existing track differs from shared prefix')
        new=obj['events'][len(j.saved):]
        if new:
            with j.db:
                for i,source in enumerate(new,len(j.saved)):
                    rec=copy.deepcopy(source);rec['result']['cache_hit']=True;rec['result']['physical_training_seconds']=0.
                    j.db.execute('INSERT INTO calls VALUES (?,?,?)',(i,json.dumps(rec),hash_json(rec)))
                j.db.execute("UPDATE meta SET value=? WHERE key='committed_fe'",(str(obj['fe']),))
            j.saved=read_events(j.path)
        prefix=j.saved[:len(obj['events'])]
        j.seq=len(prefix);j.fes=obj['fe'];j.seen={r['candidate_id']:r for r in prefix if r['new_fe']}
        candidates=[r for r in prefix if r['new_fe'] and r['result']['status']=='complete']
        j.best=max(candidates,key=lambda r:(r['result']['f1'],-r['result']['loss'])) if candidates else None
        j.last_pruned=False
        restore_state(c,obj['state'])

        if SWITCH[c.variant] is None:c.switch_fe_actual=c.switch_generation=None
        atomic_json(j.path.parent/'prefix_reuse.json',dict(source=str(path),source_variant=obj['source_variant'],
            reused_fe=obj['fe'],newly_imported_fe=sum(r['new_fe'] for r in new),checksum=obj['checksum']))
        return obj['fe']
    return 0

def publish(c,j,args):

    committed=int(j.db.execute("SELECT value FROM meta WHERE key='committed_fe'").fetchone()[0])
    if j.fes<committed:return
    state=export_state(c);events=read_events(j.path)[:j.seq]
    obj=dict(identity=dict(protocol_id=args.protocol_id,run=args.run,seed=args.seed,budget=args.max_fe,
        engine_sha256=runtime_hashes()['hga_optimizer.py']),source_variant=c.variant,fe=j.fes,state=state,events=events)
    obj['checksum']=digest(obj)
    for target in VARIANTS:
        if not compatible(state,target,args.max_fe):continue
        path=Path(args.output)/'shared_prefix'/f'run_{args.run:02d}'/f'{c.variant}_to_{target}.json'
        if path.exists() and read_json(path)['fe']>j.fes:continue
        if args.mode=='atr' and j.best:
            SharedWeights(args).pin(weight_identity(args,args.run,j.best),'prefix:'+str(path.relative_to(Path(args.output))))
        atomic_json(path,obj)

def attach(c,j,args):
    try:
        apply(c,j,args)
    except (ValueError,RuntimeError,OSError,KeyError,TypeError) as exc:

        j.saved=read_events(j.path)
        j.seq=j.fes=0;j.seen={};j.best=None;j.last_pruned=False
        fresh=ControlledHGA(c.variant,j,args.seed+args.run,args.max_fe)
        c.__dict__.update(fresh.__dict__)
        args.independent_reuse=True
        atomic_json(j.path.parent/'independent_reuse_fallback.json',dict(stage='HGA prefix',reason=str(exc),
            action='Replay own checked journal and continue with metric/weight/prefix sharing disabled.'))
    def boundary(controller):
        if not getattr(args,'independent_reuse',False):
            try:publish(controller,j,args)
            except (ValueError,RuntimeError,OSError,KeyError,TypeError) as exc:
                args.independent_reuse=True
                atomic_json(j.path.parent/'independent_reuse_fallback.json',dict(stage='HGA shared publication',reason=str(exc),action='Continue own current optimizer state without sharing.'))
        if j.fes>=j.target:raise Paused()
    c.boundary_hook=boundary
