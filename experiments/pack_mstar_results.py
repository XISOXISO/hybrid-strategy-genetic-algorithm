from pathlib import Path as _Path
import sys as _sys
_experiment_dir = _Path(__file__).resolve().parent
for _search_path in (_experiment_dir.parent, _experiment_dir):
    if str(_search_path) not in _sys.path:
        _sys.path.insert(0, str(_search_path))
import argparse
from pathlib import Path
import zipfile
from mstar_common import FileLock, file_hash, read_json
from report_mstar import collect

def pack(output, archive, include_weights=False):
    output = Path(output).resolve()
    archive = Path(archive).resolve()
    root = Path(__file__).parent.resolve()
    if output == archive or output in archive.parents:
        raise ValueError('ZIP must be outside results directory')
    with FileLock(output / 'experiment.lock'):
        from run_mstar import count_fe
        for path in output.glob('runs/*/*/journal.sqlite'):
            count_fe(path)
        benchmark, tracks, _ = collect([output])
        from mstar_final_evaluation import validate_test_result
        if include_weights:
            from types import SimpleNamespace
            from mstar_checkpoint_archive import validate_entry
            protocol = read_json(output / 'protocol.json')
            for (name, run), rows in tracks.items():
                rd = output / 'runs' / name / f'run_{run:02d}'
                args = SimpleNamespace(protocol_id=protocol['protocol_id'], dataset_id=benchmark['dataset_id'], algorithm=name, run=run, max_fe=benchmark['max_fe'])
                for index in (rd / 'budget_checkpoints').glob('fe_*.json'):
                    validate_entry(rd, index, args, rows)
        for name, run in tracks:
            directory = output / 'runs' / name / f'run_{run:02d}'
            validate_test_result(directory)
            if include_weights and (directory / 'best_model.pt').exists():
                meta = read_json(directory / 'best_model.json')
                if file_hash(directory / 'best_model.pt') != meta['weights_sha256']:
                    raise ValueError('Best model file hash differs')
        files = []
        for f in root.iterdir():
            if f.is_file() and f.suffix in {'.py', '.sh', '.md', '.txt', '.json'}:
                files.append((f, 'code/' + f.name))
        for f in (root / 'v1_reference').rglob('*'):
            if f.is_file() and f.suffix in {'.py', '.json'}:
                files.append((f, 'code/' + f.relative_to(root).as_posix()))
        for f in output.iterdir():
            if f.is_file() and f.suffix in {'.json', '.log', '.csv', '.jsonl'}:
                files.append((f, 'results/' + f.name))
        for name in ['reports', 'paper_outputs', 'runs', 'shared_prefix', *(('shared_weights',) if include_weights else ())]:
            for f in (output / name).rglob('*'):
                if f.is_file() and (f.suffix in {'.json', '.jsonl', '.csv', '.log', '.sqlite', '.png', '.pdf'} or (include_weights and f.name == 'best_model.pt')):
                    files.append((f, 'results/' + f.relative_to(output).as_posix()))
        archive.parent.mkdir(parents=True, exist_ok=True)
        staging = archive.with_name(archive.name + '.building')
        try:
            with zipfile.ZipFile(staging, 'w', zipfile.ZIP_DEFLATED, compresslevel=6) as z:
                for f, name in sorted(files, key=lambda item: item[1]):
                    z.write(f, name)
            staging.replace(archive)
        finally:
            staging.unlink(missing_ok=True)
    return archive

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--results', default='results/formal_runs_0_9')
    p.add_argument('--output', default='formal_results.zip')
    p.add_argument('--include-weights', action='store_true', help='Optional best_model.pt files; default is reports/journals only')
    a = p.parse_args()
    out = pack(a.results, a.output, a.include_weights)
    print(out)
    print(f'{out.stat().st_size / 1024 ** 2:.2f} MiB; no dataset images; weights only with --include-weights')
if __name__ == '__main__':
    main()
