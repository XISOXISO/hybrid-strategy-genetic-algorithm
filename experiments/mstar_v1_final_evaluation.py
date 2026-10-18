from __future__ import annotations
from pathlib import Path as _Path
import sys as _sys
_experiment_dir = _Path(__file__).resolve().parent
for _search_path in (_experiment_dir.parent, _experiment_dir):
    if str(_search_path) not in _sys.path:
        _sys.path.insert(0, str(_search_path))
import csv, hashlib, math, statistics, time, traceback
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from mstar_v1_common import FileLock, atomic_csv, atomic_json, digest, file_hash, read_json
from mstar_v1_checkpoints import selection_identity, load_best
TEST_POLICY = {'version': '2.0', 'selection_budget': 500, 'model': 'validation-best epoch of validation-selected complete trial', 'training_data': '3617 train only; no train+validation refit', 'test_size': 779, 'selection_feedback': False, 'retraining': 'only if selected checkpoint is missing; exact original config/seed/epochs'}

def load_test(folder, expected_dataset_id, exact_default=True):
    from mstar_v1_dataset import validate_manifest, check_image
    folder = Path(folder)
    data = validate_manifest(read_json(folder / 'dataset.json'), expected_dataset_id, exact_default)
    rows = [r for r in data['identity']['samples'] if r['split'] == 'test']
    for row in rows:
        check_image(folder, row)
    return (data, rows)

def metrics_from_predictions(rows, predictions, classes):
    if len(rows) != len(predictions):
        raise ValueError('Prediction count differs')
    count = len(classes)
    conf = np.zeros((count, count), dtype=np.int64)
    for row, p in zip(rows, predictions):
        if not 0 <= int(p) < count:
            raise ValueError('Invalid predicted label')
        conf[int(row['label']), int(p)] += 1
    tp = np.diag(conf).astype(float)
    support = conf.sum(1)
    den = conf.sum(0) + support
    f1 = np.divide(2 * tp, den, out=np.zeros(count), where=den != 0)
    return ({'test_macro_f1': float(f1.mean()), 'test_accuracy': float(tp.sum() / max(1, conf.sum())), 'test_samples': len(rows), 'per_class_f1': f1.tolist(), 'per_class_support': support.tolist()}, conf)

def validate_test_result(directory, expected=None, expected_rows=None):
    directory = Path(directory)
    path = directory / 'test_metrics.json'
    if not path.exists():
        return None
    result = read_json(path)
    copy = dict(result)
    checksum = copy.pop('checksum', None)
    if digest(copy) != checksum:
        raise ValueError('Test metrics checksum differs')
    if expected is not None and result['selection'] != expected:
        raise ValueError('Test result belongs to another selection')
    if result['test_policy'] != TEST_POLICY:
        raise ValueError('Test policy changed')
    for name, h in result['output_hashes'].items():
        if name not in ['test_predictions.csv', 'test_confusion_matrix.csv', 'test_confusion_matrix.png']:
            raise ValueError('Invalid test artifact path')
        if file_hash(directory / name) != h:
            raise ValueError('Test output checksum differs: ' + name)
    with (directory / 'test_predictions.csv').open(encoding='utf-8-sig', newline='') as f:
        rows = list(csv.DictReader(f))
    if len(rows) != result['test_samples']:
        raise ValueError('Stored test count differs')
    if expected_rows is not None and (len(rows) != len(expected_rows) or any((a['image'] != b['copied_image'] or int(a['true_label']) != b['label'] or a['true_class'] != b['class'] for a, b in zip(rows, expected_rows)))):
        raise ValueError('Test predictions do not follow frozen sample order/labels')
    calculated, conf = metrics_from_predictions([{'label': int(r['true_label'])} for r in rows], [int(r['predicted_label']) for r in rows], result['classes'])
    for key in ['test_macro_f1', 'test_accuracy']:
        if abs(calculated[key] - result[key]) > 1e-12:
            raise ValueError('Test metrics/predictions differ')
    with (directory / 'test_confusion_matrix.csv').open(encoding='utf-8-sig', newline='') as f:
        matrix = list(csv.DictReader(f))
    if len(matrix) != len(result['classes']) or any((r['true_class'] != c for r, c in zip(matrix, result['classes']))):
        raise ValueError('Confusion labels differ')
    actual = np.array([[int(r[c]) for c in result['classes']] for r in matrix])
    if not np.array_equal(actual, conf):
        raise ValueError('Confusion matrix/predictions differ')
    return result

def write_test_outputs(directory, selection, dataset_id, classes, rows, predictions, probabilities, weight_hash, seconds):
    directory = Path(directory)
    metrics, conf = metrics_from_predictions(rows, predictions, classes)
    records = []
    for r, p, prob in zip(rows, predictions, probabilities):
        record = {'image': r['copied_image'], 'true_label': r['label'], 'true_class': r['class'], 'predicted_label': int(p), 'predicted_class': classes[int(p)], 'confidence': float(prob[int(p)])}
        record.update({f'probability_{i}': float(v) for i, v in enumerate(prob)})
        records.append(record)
    if len(records) != len(rows) or any((len(p) != len(classes) for p in probabilities)):
        raise ValueError('Incomplete probability outputs')
    atomic_csv(directory / 'test_predictions.csv', records)
    atomic_csv(directory / 'test_confusion_matrix.csv', [{'true_class': c, **{cc: int(conf[i, j]) for j, cc in enumerate(classes)}} for i, c in enumerate(classes)])
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(10, 9))
    im = ax.imshow(conf, cmap='Blues')
    fig.colorbar(im, ax=ax)
    ax.set(xticks=range(len(classes)), yticks=range(len(classes)), xticklabels=classes, yticklabels=classes, xlabel='Predicted class', ylabel='True class', title='Independent test: ' + selection['algorithm'] + ' run ' + str(selection['run']))
    plt.setp(ax.get_xticklabels(), rotation=60, ha='right', fontsize=7)
    plt.setp(ax.get_yticklabels(), fontsize=7)
    for i in range(len(classes)):
        for j in range(len(classes)):
            ax.text(j, i, str(conf[i, j]), ha='center', va='center', fontsize=7, color='white' if conf[i, j] > conf.max() / 2 else 'black')
    fig.tight_layout()
    fig.savefig(directory / 'test_confusion_matrix.png', dpi=160)
    plt.close(fig)
    result = {**metrics, 'selection': selection, 'dataset_id': dataset_id, 'classes': classes, 'weights_sha256': weight_hash, 'test_policy': TEST_POLICY, 'test_seconds': seconds, 'output_hashes': {n: file_hash(directory / n) for n in ['test_predictions.csv', 'test_confusion_matrix.csv', 'test_confusion_matrix.png']}}
    result['checksum'] = digest(result)
    atomic_json(directory / 'test_metrics.json', result)
    return result

def infer_checkpoint(directory, identity, prepared, rows, device):
    import torch
    import mstar_v1_cnn_core as core
    payload = load_best(directory, identity)
    model = core.models.resnet18(weights=None)
    model.fc = core.nn.Linear(model.fc.in_features, 10)
    model.load_state_dict(payload['state_dict'], strict=True)
    model = core._model_to_device(model, device)
    model.eval()
    _, tf, _ = core.build_transforms(1.0)
    samples = [(str(Path(prepared) / r['copied_image']), r['label']) for r in rows]
    loader = core.make_loader(core.MSTARImageDataset(samples, tf, None), identity['config']['batch_size'], False, 0, device, identity['training_seed'])
    predictions = []
    probabilities = []
    start = time.monotonic()
    criterion = core.nn.CrossEntropyLoss()
    with torch.inference_mode():
        for x, y in loader:
            x = core._maybe_channels_last(x.to(device, non_blocking=True))
            y = y.to(device, non_blocking=True)
            if device.type == 'cuda':
                logits, loss = core._bf16_forward_fp32_loss(model, x, y, criterion)
            else:
                logits = model(x)
                loss = criterion(logits.float(), y)
            if not bool(torch.isfinite(logits).all()) or not bool(torch.isfinite(loss)):
                raise RuntimeError('Nonfinite test inference')
            prob = logits.float().softmax(1).cpu().numpy()
            predictions.extend(prob.argmax(1).tolist())
            probabilities.extend(prob.tolist())
    return (predictions, probabilities, time.monotonic() - start)

def test_worker(task):
    from mstar_v1_backend import MSTARBackend
    from mstar_v1_journal import read_events
    from mstar_v1_reporting import selected
    a = SimpleNamespace(**task)
    rd = Path(a.output) / 'runs' / a.algorithm / f'run_{a.run:02d}'
    with FileLock(rd / 'run.lock'):
        try:
            events = read_events(rd / 'journal.sqlite')
            unique = [r for r in events if r['new_fe']]
            if a.smoke_epochs or a.max_fe != 500 or len(unique) != 500:
                raise ValueError('Independent test requires completed MAXFE500')
            best = selected(unique, 500)
            if best is None:
                raise ValueError('No complete eligible selected model')
            b = MSTARBackend(a, a.run, rd)
            b.journal = SimpleNamespace(fes=500, best=best, seen={r['candidate_id']: r for r in unique})
            try:
                b.ensure_checkpoint(best)
            finally:
                b.close()
            identity = selection_identity(a, best)
            data, rows = load_test(a.data, a.dataset_id)
            old = validate_test_result(rd, identity, rows)
            if old:
                if old['dataset_id'] != a.dataset_id or old['weights_sha256'] != file_hash(rd / 'best_model.pt'):
                    raise ValueError('Test/model identity changed')
                atomic_json(rd / 'status.json', {'state': 'test_complete', 'completed_fe': 500, 'test_macro_f1': old['test_macro_f1']})
                return old
            atomic_json(rd / 'status.json', {'state': 'testing', 'completed_fe': 500})
            import torch
            import mstar_v1_cnn_core as core
            core.configure_performance(SimpleNamespace(performance_mode='safe', prefetch_factor=2, val_cache='auto', val_gpu_fraction=0.18, ram_image_cache='on', clear_cuda_cache_every=0))
            torch.use_deterministic_algorithms(True)
            core.set_seed(best['result']['training_seed'])
            predictions, probabilities, seconds = infer_checkpoint(rd, identity, a.data, rows, torch.device('cuda:0'))
            result = write_test_outputs(rd, identity, a.dataset_id, data['identity']['classes'], rows, predictions, probabilities, file_hash(rd / 'best_model.pt'), seconds)
            atomic_json(rd / 'status.json', {'state': 'test_complete', 'completed_fe': 500, 'test_macro_f1': result['test_macro_f1']})
            return result
        except BaseException as exc:
            with (rd / 'test_error.log').open('a') as log:
                traceback.print_exc(file=log)
            atomic_json(rd / 'status.json', {'state': 'test_error', 'completed_fe': 500, 'error': str(exc)})
            raise

def report_tests(outputs, destination, tracks, benchmark, formal):
    results = {}
    progress = []
    for output in map(Path, outputs):
        protocol = read_json(output / 'protocol.json')
        for name in protocol['identity']['algorithms']:
            for run in protocol['selected_runs']:
                rd = output / 'runs' / name / f'run_{run:02d}'
                result = validate_test_result(rd)
                if result:
                    from mstar_v1_reporting import selected
                    best = selected(tracks[name, run], 500)
                    args = SimpleNamespace(protocol_id=protocol['protocol_id'], dataset_id=benchmark['dataset_id'], algorithm=name, run=run)
                    if len(tracks[name, run]) != 500 or best is None or result['selection'] != selection_identity(args, best):
                        raise ValueError('Test selection does not match frozen FE500 journal')
                    if result['dataset_id'] != benchmark['dataset_id'] or result['test_samples'] != 779:
                        raise ValueError('Test dataset/count differs')
                    from mstar_v1_dataset import validate_manifest
                    snapshot = validate_manifest(read_json(output / 'frozen_dataset_snapshot.json'), benchmark['dataset_id'])
                    expected_rows = [r for r in snapshot['identity']['samples'] if r['split'] == 'test']
                    if result['classes'] != snapshot['identity']['classes']:
                        raise ValueError('Test class mapping changed')
                    validate_test_result(rd, result['selection'], expected_rows)
                    results[name, run] = result
                progress.append({'algorithm': name, 'run': run, 'test_complete': result is not None, 'test_macro_f1': result['test_macro_f1'] if result else None, 'test_accuracy': result['test_accuracy'] if result else None})
    destination = Path(destination)
    atomic_csv(destination / 'test_task_progress.csv', progress)
    summaries = []
    names = sorted({n for n, r in tracks})
    for n in names:
        values = [v['test_macro_f1'] for (name, r), v in results.items() if name == n]
        summaries.append({'algorithm': n, 'tested_runs': len(values), 'mean_test_macro_f1': statistics.mean(values) if values else None, 'sd_test_macro_f1': statistics.stdev(values) if len(values) > 1 else None, 'mean_test_accuracy': statistics.mean([v['test_accuracy'] for (name, r), v in results.items() if name == n]) if values else None, 'min': min(values) if values else None, 'max': max(values) if values else None})
    atomic_csv(destination / 'test_summary.csv', summaries)
    complete = benchmark['mode'] == 'mstar' and len(results) == len(tracks) and bool(tracks)
    stats = {'performed': False, 'reason': 'Requires all 10 runs, MAXFE500 and all test results for present algorithms'}
    if formal and complete and (len(names) >= 3):
        from scipy.stats import friedmanchisquare, wilcoxon, rankdata
        import itertools, warnings
        values = {n: [results[n, r]['test_macro_f1'] for r in range(10)] for n in names}
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            f = friedmanchisquare(*(values[n] for n in names))
        pairs = []
        for a, b in itertools.combinations(names, 2):
            diff = [x - y for x, y in zip(values[a], values[b])]
            p = 1.0 if not any(diff) else float(wilcoxon(values[a], values[b], alternative='two-sided').pvalue)
            pairs.append({'algorithm_a': a, 'algorithm_b': b, 'mean_paired_f1_difference': statistics.mean(diff), 'raw_p': p})
        previous = 0.0
        for rank, i in enumerate(sorted(range(len(pairs)), key=lambda i: pairs[i]['raw_p'])):
            previous = max(previous, min(1.0, (len(pairs) - rank) * pairs[i]['raw_p']))
            pairs[i]['holm_p'] = previous
        ranks = [rankdata([-values[n][r] for n in names], method='average') for r in range(10)]
        atomic_csv(destination / 'test_average_rank.csv', [{'algorithm': n, 'mean_rank': float(np.mean([v[i] for v in ranks]))} for i, n in enumerate(names)])
        atomic_csv(destination / 'test_wilcoxon_holm.csv', pairs)
        stats = {'performed': True, 'paired_runs': 10, 'friedman_statistic': float(f.statistic) if math.isfinite(f.statistic) else None, 'friedman_p': float(f.pvalue) if math.isfinite(f.pvalue) else None, 'holm_family_size': len(pairs)}
    atomic_json(destination / 'test_statistics.json', stats)
    return {'test_evaluated': bool(results), 'tested_tracks': len(results), 'test_complete_for_present_tracks': complete, 'formal_pipeline_complete_for_present_algorithms': formal and complete}
