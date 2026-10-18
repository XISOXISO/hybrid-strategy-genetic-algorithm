from __future__ import annotations
from pathlib import Path as _Path
import sys as _sys
_experiment_dir = _Path(__file__).resolve().parent
for _search_path in (_experiment_dir.parent, _experiment_dir):
    if str(_search_path) not in _sys.path:
        _sys.path.insert(0, str(_search_path))
import argparse, csv, math, statistics, warnings
from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from mstar_common import FileLock, atomic_csv, atomic_json, digest, file_hash, read_json
from src.HGA.mstar.checkpoints import selection_identity
from report_mstar import collect, selected, report
from mstar_run_audit import timing_summary
from mstar_final_evaluation import validate_test_result, metrics_from_predictions
PREFIX_POLICY = {'version': 1, 'purpose': 'post-search fixed-budget descriptive test curve', 'selection': 'best validation macro-F1 then lowest validation loss within fixed FE budget', 'selection_feedback': False, 'retraining': False, 'training_data': 'original training split only'}

def interval(values, bounded=True):
    values = [float(v) for v in values]
    n = len(values)
    if not n:
        return {'n': 0, 'mean': None, 'sd': None, 'ci95_low': None, 'ci95_high': None}
    mean = statistics.mean(values)
    sd = statistics.stdev(values) if n > 1 else None
    low = high = None
    if n > 1:
        from scipy.stats import t
        margin = float(t.ppf(0.975, n - 1)) * sd / math.sqrt(n)
        low = mean - margin
        high = mean + margin
        if bounded:
            low = max(0.0, low)
            high = min(1.0, high)
    return {'n': n, 'mean': mean, 'sd': sd, 'ci95_low': low, 'ci95_high': high}

def resource_summary(rd):
    path = Path(rd) / 'resource_samples.csv'
    empty = {'resource_samples': 0, 'sampled_seconds': 0.0, **{key + '_mean': None for key in ['cpu_cores_percent', 'cpu_allocation_percent', 'rss_mib', 'process_vram_mib', 'device_gpu_percent', 'device_vram_percent']}, 'shared_device_samples': 0, 'gpu_mean_scope': 'whole device during track activity; shared samples cannot rank algorithm GPU demand', 'sampling_error_samples': 0}
    if not path.exists():
        return empty
    with path.open(encoding='utf-8', newline='') as f:
        rows = [r for r in csv.DictReader(f) if r['stage'] == 'search']
    valid = [r for r in rows if r['sample_interval_seconds']]

    def average(key):
        subset = [r for r in valid if r.get(key) not in ('', None)]
        total = sum((float(r['sample_interval_seconds']) for r in subset))
        return sum((float(r[key]) * float(r['sample_interval_seconds']) for r in subset)) / total if total else None
    seconds = sum((float(r['sample_interval_seconds']) for r in valid))
    return {'resource_samples': len(rows), 'sampled_seconds': seconds, **{key + '_mean': average(key) for key in ['cpu_cores_percent', 'cpu_allocation_percent', 'rss_mib', 'process_vram_mib', 'device_gpu_percent', 'device_vram_percent']}, 'shared_device_samples': sum((r['gpu_scope'] == 'shared_device' for r in rows)), 'gpu_mean_scope': empty['gpu_mean_scope'], 'sampling_error_samples': sum((bool(r['sampling_error']) for r in rows))}

def validate_prefix(folder, meta, rows, classes):
    path = folder / 'offline_test_metrics.json'
    if not path.exists():
        return None
    r = read_json(path)
    c = dict(r)
    checksum = c.pop('checksum', None)
    if checksum != digest(c) or r['selection'] != meta['selection'] or r['weights_sha256'] != meta['weights_sha256'] or (r['test_policy'] != PREFIX_POLICY):
        raise ValueError('Offline test selection/checksum mismatch')
    for name, h in r['output_hashes'].items():
        if name not in ['offline_test_predictions.csv', 'offline_test_confusion_counts.csv'] or file_hash(folder / name) != h:
            raise ValueError('Offline test artifact checksum mismatch')
    with (folder / 'offline_test_predictions.csv').open(encoding='utf-8-sig', newline='') as f:
        pred = list(csv.DictReader(f))
    if len(pred) != len(rows) or any((p['image'] != s['copied_image'] or int(p['true_label']) != s['label'] for p, s in zip(pred, rows))):
        raise ValueError('Offline test sample order differs')
    calculated, _ = metrics_from_predictions(rows, [int(p['predicted_label']) for p in pred], classes)
    if any((calculated[k] != r[k] for k in calculated)):
        raise ValueError('Offline test predictions/metrics differ')
    return r

def evaluate_archives(outputs, data_folder, device='cuda:0'):
    import torch
    from mstar_final_evaluation import load_test, infer_checkpoint
    from mstar_checkpoint_archive import validate_entry
    b, tracks, sources = collect(outputs)
    if b['mode'] != 'mstar' or b.get('smoke_only') or b['max_fe'] != 500 or any((len(v) != 500 for v in tracks.values())):
        raise ValueError('Offline test curves require all present formal search selections frozen at FE500')
    data, rows = load_test(data_folder, b['dataset_id'])
    classes = data['identity']['classes']
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    for (name, run), records in sorted(tracks.items()):
        output = Path(sources[name, run])
        p = read_json(output / 'protocol.json')
        rd = output / 'runs' / name / f'run_{run:02d}'
        args = SimpleNamespace(protocol_id=p['protocol_id'], dataset_id=b['dataset_id'], algorithm=name, run=run, max_fe=500)
        for index in sorted((rd / 'budget_checkpoints').glob('fe_*.json')):
            meta, obj = validate_entry(rd, index, args, records)
            if meta['status'] != 'available':
                continue
            if validate_prefix(obj, meta, rows, classes):
                continue
            predictions, probabilities, seconds = infer_checkpoint(obj, meta['selection'], data_folder, rows, torch.device(device))
            metrics, conf = metrics_from_predictions(rows, predictions, classes)
            atomic_csv(obj / 'offline_test_predictions.csv', [{'image': r['copied_image'], 'true_label': r['label'], 'predicted_label': int(pred), 'confidence': float(prob[int(pred)]), **{f'probability_{i}': float(v) for i, v in enumerate(prob)}} for r, pred, prob in zip(rows, predictions, probabilities)])
            atomic_csv(obj / 'offline_test_confusion_counts.csv', [{'true_class': c, **{cc: int(conf[i, j]) for j, cc in enumerate(classes)}} for i, c in enumerate(classes)])
            result = {**metrics, 'classes': classes, 'dataset_id': b['dataset_id'], 'selection': meta['selection'], 'test_policy': PREFIX_POLICY, 'weights_sha256': meta['weights_sha256'], 'test_seconds': seconds, 'output_hashes': {n: file_hash(obj / n) for n in ['offline_test_predictions.csv', 'offline_test_confusion_counts.csv']}}
            result['checksum'] = digest(result)
            atomic_json(obj / 'offline_test_metrics.json', result)
            print(f"Offline test: {name} run {run} FE{meta['budget_fe']} = {metrics['test_macro_f1']:.6f}", flush=True)

def generate(outputs, destination, plots=True):
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    b, tracks, sources = collect(outputs)
    names = sorted({n for n, r in tracks})
    final = {}
    timings = []
    archive_rows = []
    class_rows = []
    report(outputs, destination / 'standard_reports', plots=plots)
    from mstar_checkpoint_archive import validate_entry
    for (name, run), rows in sorted(tracks.items()):
        output = Path(sources[name, run])
        p = read_json(output / 'protocol.json')
        rd = output / 'runs' / name / f'run_{run:02d}'
        args = SimpleNamespace(protocol_id=p['protocol_id'], dataset_id=b['dataset_id'], algorithm=name, run=run, max_fe=b['max_fe'])
        timing = timing_summary(rd)
        resources = resource_summary(rd)
        timings.append({'algorithm': name, 'run': run, 'completed_fe': len(rows), 'max_fe': b['max_fe'], **timing, **resources, 'cache_hits': sum((r['result'].get('cache_hit', False) for r in rows)), 'recorded_successful_physical_training_seconds': sum((r['result']['physical_training_seconds'] for r in rows))})
        result = validate_test_result(rd)
        if result:
            best = selected(rows, b['max_fe'])
            if len(rows) != 500 or result['selection'] != selection_identity(args, best) or result['dataset_id'] != b['dataset_id'] or ((rd / 'best_model.pt').exists() and result['weights_sha256'] != file_hash(rd / 'best_model.pt')):
                raise ValueError('Final test does not match frozen selected weights')
            final[name, run] = result
        for index in sorted((rd / 'budget_checkpoints').glob('fe_*.json')):
            meta, obj = validate_entry(rd, index, args, rows, require_weights=False)
            value = None
            if meta['status'] == 'available' and (obj / 'offline_test_metrics.json').exists():
                r = read_json(obj / 'offline_test_metrics.json')
                check = dict(r)
                checksum = check.pop('checksum', None)
                if checksum != digest(check) or r['selection'] != meta['selection'] or r['weights_sha256'] != meta['weights_sha256'] or (r['dataset_id'] != b['dataset_id']) or (r['test_policy'] != PREFIX_POLICY):
                    raise ValueError('Offline test metrics identity mismatch')
                for fn, h in r['output_hashes'].items():
                    if fn not in ['offline_test_predictions.csv', 'offline_test_confusion_counts.csv'] or file_hash(obj / fn) != h:
                        raise ValueError('Offline test output checksum mismatch')
                value = r['test_macro_f1']
            archive_rows.append({'algorithm': name, 'run': run, 'budget_fe': meta['budget_fe'], 'checkpoint_status': 'weights_omitted_or_missing' if meta['status'] == 'available' and (not (obj / 'best_model.pt').exists()) else meta['status'], 'selected_fe': meta['selection']['selected_fe'], 'validation_macro_f1': meta['selection']['validation_macro_f1'], 'test_macro_f1': value})
    atomic_csv(destination / 'timing_resource_runs.csv', timings)
    atomic_csv(destination / 'fixed_budget_runs.csv', archive_rows, ['algorithm', 'run', 'budget_fe', 'checkpoint_status', 'selected_fe', 'validation_macro_f1', 'test_macro_f1'])
    summary = []
    for name in names:
        available = [v for (n, r), v in sorted(final.items()) if n == name]
        s = {'algorithm': name, 'expected_global_runs': b['runs'], 'present_search_runs': sum((n == name for n, r in tracks)), 'final_test_runs': len(available), **{'test_f1_' + k: v for k, v in interval([v['test_macro_f1'] for v in available]).items()}}
        full = [t for t in timings if t['algorithm'] == name and t['completed_fe'] == b['max_fe'] and t['timing_complete_from_fe0']]
        s.update({'complete_measured_' + k: v for k, v in interval([t['observed_active_wall_seconds'] for t in full], False).items()})
        s['timing_comparison_note'] = 'Shared-cache/concurrent timings describe this execution; fair comparisons require cold isolated runs'
        summary.append(s)
        if available:
            for i, cls in enumerate(available[0]['classes']):
                class_rows.append({'algorithm': name, 'class': cls, **interval([v['per_class_f1'][i] for v in available])})
            matrices = []
            for (n, r), v in sorted(final.items()):
                if n != name:
                    continue
                rd = Path(sources[n, r]) / 'runs' / n / f'run_{r:02d}'
                with (rd / 'test_confusion_matrix.csv').open(encoding='utf-8-sig', newline='') as f:
                    cm = list(csv.DictReader(f))
                matrices.append([[int(row[c]) for c in v['classes']] for row in cm])
            counts = np.sum(matrices, axis=0)
            support = counts.sum(1, keepdims=True)
            percent = np.divide(counts * 100.0, support, out=np.zeros_like(counts, dtype=float), where=support != 0)
            atomic_csv(destination / f'{name}_confusion_percent.csv', [{'true_class': c, **{cc: float(percent[i, j]) for j, cc in enumerate(available[0]['classes'])}} for i, c in enumerate(available[0]['classes'])])
            if plots:
                import matplotlib
                matplotlib.use('Agg')
                import matplotlib.pyplot as plt
                fig, ax = plt.subplots(figsize=(9, 8))
                im = ax.imshow(percent, vmin=0, vmax=100, cmap='Blues')
                fig.colorbar(im, ax=ax, label='%')
                ax.set(xticks=range(10), yticks=range(10), xticklabels=available[0]['classes'], yticklabels=available[0]['classes'], xlabel='Predicted', ylabel='True', title=f'{name}: pooled {len(available)} runs')
                plt.setp(ax.get_xticklabels(), rotation=60, ha='right', fontsize=7)
                for i in range(10):
                    for j in range(10):
                        ax.text(j, i, f'{percent[i, j]:.1f}', ha='center', va='center', fontsize=6, color='white' if percent[i, j] > 50 else 'black')
                fig.tight_layout()
                fig.savefig(destination / f'{name}_confusion_percent.png', dpi=180)
                plt.close(fig)
    atomic_csv(destination / 'final_test_ci95.csv', summary)
    atomic_csv(destination / 'per_class_f1_ci95.csv', class_rows, ['algorithm', 'class', 'n', 'mean', 'sd', 'ci95_low', 'ci95_high'])
    paired = sorted((r for n, r in final if n == 'JSO' and ('LSHADE', r) in final))
    formal = b['mode'] == 'mstar' and (not b.get('smoke_only')) and (set(paired) == set(range(10)))
    stats = {'performed': False, 'reason': 'Requires all 10 paired formal final tests', 'paired_run_ids': paired, 'friedman_performed': False, 'friedman_reason': 'Only two algorithms; use a paired Wilcoxon test', 'ci_method': 'Student t over independent RUN scores; F1 endpoints clipped to [0,1]'}
    if formal:
        from scipy.stats import wilcoxon
        x = [final['JSO', r]['test_macro_f1'] for r in paired]
        y = [final['LSHADE', r]['test_macro_f1'] for r in paired]
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            p = 1.0 if x == y else float(wilcoxon(x, y, alternative='two-sided').pvalue)
        stats.update(performed=True, raw_p=p, holm_p=p, holm_family_size=1, mean_paired_difference=statistics.mean((a - c for a, c in zip(x, y))), alternative='two-sided')
    atomic_json(destination / 'paired_statistics.json', stats)
    curve = []
    for name in names:
        for fe in sorted({r['budget_fe'] for r in archive_rows}):
            found = [r for r in archive_rows if r['algorithm'] == name and r['budget_fe'] == fe]
            observed = [r['test_macro_f1'] for r in found if r['test_macro_f1'] is not None]
            curve.append({'algorithm': name, 'budget_fe': fe, 'present_runs': sum((n == name for n, r in tracks)), 'available_checkpoint_runs': sum((r['checkpoint_status'] == 'available' for r in found)), **interval(observed)})
    atomic_csv(destination / 'offline_test_curve_ci95.csv', curve, ['algorithm', 'budget_fe', 'present_runs', 'available_checkpoint_runs', 'n', 'mean', 'sd', 'ci95_low', 'ci95_high'])
    if plots:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        plotted = [s for s in summary if s['test_f1_mean'] is not None]
        if plotted:
            fig, ax = plt.subplots(figsize=(6, 4))
            means = [s['test_f1_mean'] for s in plotted]
            ax.bar([s['algorithm'] for s in plotted], means)
            for i, s in enumerate(plotted):
                if s['test_f1_ci95_low'] is not None:
                    ax.errorbar(i, means[i], yerr=[[means[i] - s['test_f1_ci95_low']], [s['test_f1_ci95_high'] - means[i]]], color='black', capsize=4)
            ax.set(ylabel='Final test macro-F1', ylim=(0, 1.01), title='ResNet18 / FE500; 95% RUN-level CI')
            fig.tight_layout()
            fig.savefig(destination / 'final_test_ci95.png', dpi=180)
            plt.close(fig)
        if any((r['n'] for r in curve)):
            fig, ax = plt.subplots(figsize=(8, 4))
            for name in names:
                available = [r for r in curve if r['algorithm'] == name and r['n'] == r['present_runs'] and r['n']]
                if not available:
                    continue
                ax.plot([r['budget_fe'] for r in available], [r['mean'] for r in available], marker='o', label=name)
            ax.set(xlabel='Fixed unique FE budget', ylabel='Post-search test macro-F1', ylim=(0, 1.01), title='Frozen validation selections; only complete present-run checkpoints shown')
            ax.legend()
            fig.tight_layout()
            fig.savefig(destination / 'offline_test_curve.png', dpi=180)
            plt.close(fig)
    coverage = {'algorithms': names, 'backbone': 'ResNet18', 'max_fe': b['max_fe'], 'expected_runs': b['runs'], 'formal_paired_test_complete': formal, 'missing_historical_weights': sum((r['checkpoint_status'] == 'historical_weights_missing' for r in archive_rows)), 'omitted_or_missing_recorded_weights': sum((r['checkpoint_status'] == 'weights_omitted_or_missing' for r in archive_rows)), 'timing_resource_policy': 'Historical gaps stay missing. GPU utilization is device-wide. Sampling is approximate.', 'interpretation': 'Partial-shard CIs are descriptive; formal paired significance requires all 10 RUNs.'}
    atomic_json(destination / 'coverage.json', coverage)
    return coverage

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('outputs', nargs='+')
    p.add_argument('--destination', required=True)
    p.add_argument('--evaluate-checkpoints', action='store_true')
    p.add_argument('--data', default=str(Path(__file__).parent / 'data'))
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--no-plots', action='store_true')
    a = p.parse_args()
    with ExitStack() as stack:
        for output in sorted(set((str(Path(o).resolve()) for o in a.outputs))):
            stack.enter_context(FileLock(Path(output) / 'experiment.lock'))
        if a.evaluate_checkpoints:
            evaluate_archives(a.outputs, a.data, a.device)
        print(generate(a.outputs, a.destination, not a.no_plots))
if __name__ == '__main__':
    main()
