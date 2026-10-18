from pathlib import Path as _Path
import sys as _sys
_experiment_dir = _Path(__file__).resolve().parent
for _search_path in (_experiment_dir.parent, _experiment_dir):
    if str(_search_path) not in _sys.path:
        _sys.path.insert(0, str(_search_path))
import argparse, shutil, sqlite3
from pathlib import Path
from mstar_common import FileLock, atomic_json, parse_runs, read_json
from report_mstar import collect

def export_runs(source, destination, ids):
    source = Path(source).resolve()
    destination = Path(destination).resolve()
    if source == destination or source in destination.parents:
        raise ValueError('New output must be outside source')
    if destination.exists():
        raise ValueError('Destination already exists; use a new directory')
    with FileLock(source / 'experiment.lock'):
        protocol = read_json(source / 'protocol.json')
        if not set(ids) <= set(protocol['selected_runs']):
            raise ValueError('Requested run not present')
        from run_mstar import count_fe
        for path in source.glob('runs/*/*/journal.sqlite'):
            count_fe(path)
        collect([source])
        staging = destination.with_name(destination.name + '.copying')
        if staging.exists():
            raise ValueError('Staging directory already exists')
        staging.mkdir(parents=True)
        try:
            for name in ('frozen_dataset_snapshot.json', 'data_audit_snapshot.json', 'hardware.json', 'upgrade_import.json', 'device_resources.csv'):
                if (source / name).exists():
                    shutil.copy2(source / name, staging / name)
            if (source / 'shared_weights').exists():
                shutil.copytree(source / 'shared_weights', staging / 'shared_weights')
            for run in ids:
                for name in protocol['identity']['algorithms']:
                    relative = Path('runs') / name / f'run_{run:02d}'
                    shutil.copytree(source / relative, staging / relative, ignore=shutil.ignore_patterns('*.lock', '*.tmp', '*-journal'))
                    src = sqlite3.connect(source / relative / 'journal.sqlite')
                    dst = sqlite3.connect(staging / relative / 'journal.sqlite')
                    try:
                        src.backup(dst)
                    finally:
                        dst.close()
                        src.close()
                prefix = source / 'shared_prefix' / f'run_{run:02d}'
                if prefix.exists():
                    shutil.copytree(prefix, staging / 'shared_prefix' / prefix.name)
            protocol['selected_runs'] = sorted(ids)
            atomic_json(staging / 'protocol.json', protocol)
            collect([staging])
            staging.replace(destination)
        except BaseException:
            shutil.rmtree(staging, ignore_errors=True)
            raise
    return destination

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', required=True)
    p.add_argument('--destination', required=True)
    p.add_argument('--run-ids', required=True)
    a = p.parse_args()
    print(export_runs(a.source, a.destination, parse_runs(a.run_ids, 10)))
if __name__ == '__main__':
    main()
