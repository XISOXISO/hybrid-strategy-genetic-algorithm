from pathlib import Path as _Path
import sys as _sys
_experiment_dir = _Path(__file__).resolve().parent
for _search_path in (_experiment_dir.parent, _experiment_dir):
    if str(_search_path) not in _sys.path:
        _sys.path.insert(0, str(_search_path))
import argparse, time
from pathlib import Path
from atrnet_star_common import read_json
from atrnet_star_dashboard import Dashboard

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--results', required=True)
    p.add_argument('--once', action='store_true')
    a = p.parse_args()
    root = Path(a.results)
    ui = Dashboard()
    try:
        while True:
            live = read_json(root / 'live_progress.json') if (root / 'live_progress.json').exists() else {}
            schedule = read_json(root / 'scheduler.json') if (root / 'scheduler.json').exists() else {}
            total = live.get('target_fe', 0)
            done = live.get('completed_fe', 0)
            tasks = live.get('active', [])
            stage = schedule.get('stage', 'Preparing')
            if stage == 'test':
                tested = len(schedule.get('succeeded_tracks', []))
                protocol = read_json(root / 'protocol.json')
                task_count = len(protocol['identity']['algorithms']) * len(protocol['selected_runs'])
                fraction = 90 + 8 * tested / max(1, task_count)
                tasks = [f'{n} run {r}: isolated test' for n, r in schedule.get('active_tracks', [])]
            else:
                fraction = 12 + 76 * done / max(1, total) if total else 0
            summary_path = root / 'run_summary.json'
            scheduler_path = root / 'scheduler.json'
            if summary_path.exists() and (not scheduler_path.exists() or summary_path.stat().st_mtime_ns >= scheduler_path.stat().st_mtime_ns):
                summary = read_json(summary_path)
                stage = summary['state']
                if stage == 'COMPLETED':
                    fraction = 100
            age = time.time() - (root / 'scheduler.json').stat().st_mtime if (root / 'scheduler.json').exists() else None
            ui.update(fraction, stage, above=[f'Committed FE {done}/{total}', f"jobs={schedule.get('jobs', '--')} | failed {len(schedule.get('failed_tracks', []))} | snapshot age {(int(age) if age is not None else '--')}s"], below=tasks + [f"Queued {len(schedule.get('queued_tracks', []))} | retry wait {len(schedule.get('retry_waiters', []))} | cache wait {len(schedule.get('cache_waiters', []))}", str(root)], force=True)
            if a.once:
                break
            time.sleep(2)
    except KeyboardInterrupt:
        pass
    finally:
        ui.close()
if __name__ == '__main__':
    main()
