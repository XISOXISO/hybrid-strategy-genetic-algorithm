from __future__ import annotations
from pathlib import Path as _Path
import sys as _sys
_experiment_dir = _Path(__file__).resolve().parent
for _search_path in (_experiment_dir.parent, _experiment_dir):
    if str(_search_path) not in _sys.path:
        _sys.path.insert(0, str(_search_path))
import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
import numpy as np
from scipy.stats import rankdata, ranksums
from cec2022_algorithm_registry import ALGORITHMS
from cec2022_cec_evaluator import OFFICIAL_BIASES

def csv_write(path, rows, fields):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    with temporary.open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        for row in rows:
            writer.writerow({k: '' if isinstance(v, (float, np.floating)) and (not math.isfinite(v)) else v for k, v in row.items()})
    temporary.replace(path)

def normalize(values):
    values = np.asarray(values, dtype=float)
    span = np.max(values) - np.min(values)
    if not np.isfinite(span) or span <= 0:
        return None
    return (values - np.min(values)) / span

def inverse_score(quality, timing):
    denominator = 0.6 * quality + 0.4 * timing
    return np.divide(1.0, denominator, out=np.full_like(denominator, np.inf), where=denominator != 0)

def summarize(output, plots=True, mstar_path=''):
    output = Path(output).resolve()
    protocol = json.loads((output / 'protocol.json').read_text(encoding='utf-8'))
    core = protocol['core']
    expected = core['runs']
    groups = defaultdict(list)
    rows = []
    for path in sorted((output / 'runs').glob('*/*/run_*.npz')):
        with np.load(path, allow_pickle=False) as z:
            metadata = json.loads(str(z['metadata'].item()))
        if metadata['protocol_hash'] != protocol['protocol_hash']:
            raise RuntimeError(f'Protocol mismatch: {path}')
        if metadata['status'] != 'ok' or metadata['fe_used'] != metadata['budget']:
            raise RuntimeError(f'Invalid completed run: {path}')
        groups[metadata['fid'], metadata['algorithm']].append((metadata, path))
        rows.append({**metadata, 'best_error': metadata['best_value'] - OFFICIAL_BIASES[metadata['fid']]})
    fields = ['algorithm', 'fid', 'dim', 'budget', 'run', 'seed', 'fe_used', 'best_value', 'best_error', 'elapsed_seconds', 'evaluation_seconds', 'timing_eligible', 'shared_load_timing', 'pauses', 'resumes', 'replayed_fe', 'protocol_hash', 'completed_utc']
    csv_write(output / 'raw_runs.csv', rows, fields)
    statistics, lookup, missing = ([], {}, [])
    for case in core['cases']:
        fid = case['fid']
        for algorithm in ALGORITHMS:
            items = groups.get((fid, algorithm), [])
            found = {m['run'] for m, p in items}
            for run in range(expected):
                if run not in found:
                    checkpoint = output / 'checkpoints' / algorithm / f'F{fid}' / f'run_{run:03d}.npz'
                    missing.append({'algorithm': algorithm, 'fid': fid, 'run': run, 'status': 'checkpoint' if checkpoint.exists() else 'missing'})
            if not items:
                continue
            values = np.asarray([m['best_value'] for m, p in items])
            clean = np.asarray([m['elapsed_seconds'] for m, p in items if m['timing_eligible']])
            all_times = np.asarray([m['elapsed_seconds'] for m, p in items])
            record = {'algorithm': algorithm, 'fid': fid, 'dim': case['dim'], 'budget': case['budget'], 'n': len(items), 'BES': float(values.min()), 'AVG': float(values.mean()), 'STD': float(values.std(ddof=0)), 'AVG_error': float(values.mean() - OFFICIAL_BIASES[fid]), 'n_timing_clean': len(clean), 'mean_time_clean': float(clean.mean()) if len(clean) else float('nan'), 'std_time_clean': float(clean.std(ddof=0)) if len(clean) else float('nan'), 'mean_time_active_all': float(all_times.mean()), 'complete': len(items) == expected}
            statistics.append(record)
            lookup[fid, algorithm] = record
    csv_write(output / 'per_function_stats.csv', statistics, ['algorithm', 'fid', 'dim', 'budget', 'n', 'BES', 'AVG', 'STD', 'AVG_error', 'n_timing_clean', 'mean_time_clean', 'std_time_clean', 'mean_time_active_all', 'complete'])
    csv_write(output / 'missing_runs.csv', missing, ['algorithm', 'fid', 'run', 'status'])
    comparisons = []
    for fid in range(1, 13):
        for method in ('RM_HGA_T10', 'CM_HGA_T10', 'SW50_HGA_T10', 'SW60_HGA_T10'):
            left = groups.get((fid, method), [])
            if len(left) != expected:
                continue
            for baseline in ALGORITHMS:
                right = groups.get((fid, baseline), [])
                if baseline == method or len(right) != expected:
                    continue
                statistic, p = ranksums([m['best_value'] for m, _ in left], [m['best_value'] for m, _ in right])
                comparisons.append({'fid': fid, 'method': method, 'baseline': baseline, 'n_left': len(left), 'n_right': len(right), 'statistic': statistic, 'p_value_unadjusted': p})
    csv_write(output / 'wilcoxon_rank_sum.csv', comparisons, ['fid', 'method', 'baseline', 'n_left', 'n_right', 'statistic', 'p_value_unadjusted'])
    rank_rows = []
    for fid in range(1, 13):
        stats = [lookup.get((fid, a)) for a in ALGORITHMS]
        if any((s is None or not s['complete'] for s in stats)):
            continue
        ranks = rankdata([s['AVG'] for s in stats], method='average')
        for algorithm, rank in zip(ALGORITHMS, ranks):
            rank_rows.append({'fid': fid, 'algorithm': algorithm, 'rank': float(rank)})
    csv_write(output / 'function_ranks.csv', rank_rows, ['fid', 'algorithm', 'rank'])
    table21 = []
    for algorithm in ALGORITHMS:
        cases = [lookup.get((fid, algorithm)) for fid in range(1, 13)]
        quality_complete = all((s is not None and s['complete'] for s in cases))
        timing_complete = quality_complete and all((s['n_timing_clean'] == expected for s in cases))
        table21.append({'algorithm': algorithm, 'complete_quality': quality_complete, 'complete_timing': timing_complete, 'mean_cec': float(np.mean([s['AVG'] for s in cases])) if quality_complete else '', 'time_cec': float(np.mean([s['mean_time_clean'] for s in cases])) if timing_complete else '', 'mean_mr': '', 'time_mr': '', 'eff_cec': '', 'eff_mr': '', 'eff': ''})
    if all((r['complete_quality'] and r['complete_timing'] for r in table21)):
        quality = normalize([r['mean_cec'] for r in table21])
        timing = normalize([r['time_cec'] for r in table21])
        if quality is not None and timing is not None:
            for record, efficiency in zip(table21, inverse_score(quality, timing)):
                record['eff_cec'] = float(efficiency)
    if mstar_path:
        with Path(mstar_path).open(encoding='utf-8-sig', newline='') as stream:
            source = list(csv.DictReader(stream))
        if any((not {'algorithm', 'mean_mr', 'time_mr'} <= set(r) for r in source)):
            raise ValueError('MSTAR CSV requires algorithm, mean_mr, time_mr')
        external = {r['algorithm']: r for r in source}
        if len(external) != len(source):
            raise ValueError('Duplicate MSTAR algorithm rows')
        for record in table21:
            if record['algorithm'] in external:
                match = external[record['algorithm']]
                quality, timing = (float(match['mean_mr']), float(match['time_mr']))
                if not 0 <= quality <= 1 or not math.isfinite(timing) or timing < 0:
                    raise ValueError('Invalid MSTAR F1 or runtime')
                record['mean_mr'], record['time_mr'] = (quality, timing)
        if all((r['mean_mr'] != '' and r['time_mr'] != '' for r in table21)):
            timing = normalize([r['time_mr'] for r in table21])
            if timing is not None:
                quality = 1 - np.asarray([r['mean_mr'] for r in table21])
                for record, efficiency in zip(table21, inverse_score(quality, timing)):
                    record['eff_mr'] = float(efficiency)
                    if record['eff_cec'] != '':
                        record['eff'] = (record['eff_cec'] + record['eff_mr']) / 2
    csv_write(output / 'table21.csv', table21, ['algorithm', 'complete_quality', 'complete_timing', 'mean_cec', 'time_cec', 'mean_mr', 'time_mr', 'eff_cec', 'eff_mr', 'eff'])
    if plots:
        draw(groups, lookup, output, expected)
    print(f'Summary: {len(rows)} complete runs; {len(missing)} missing; interrupted timings are excluded from time_cec.', flush=True)

def draw(groups, lookup, output, expected):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    folder = output / 'figures'
    folder.mkdir(exist_ok=True)
    for number, fid in ((7, 1), (8, 4), (9, 7), (10, 11)):
        algorithms = [a for a in ALGORITHMS if (fid, a) in lookup and lookup[fid, a]['complete']]
        if not algorithms:
            continue
        fig, axes = plt.subplots(2, 2, figsize=(16, 10))
        colors = plt.get_cmap('tab20')
        finals, times, deviations, terminal_means = ([], [], [], [])
        for index, algorithm in enumerate(algorithms):
            trajectories = []
            for metadata, path in groups[fid, algorithm]:
                with np.load(path, allow_pickle=False) as z:
                    trajectories.append(z['best_values'].copy())
            trajectories = np.asarray(trajectories)
            mean, std = (trajectories.mean(0), trajectories.std(0, ddof=0))
            fe = np.arange(1, mean.size + 1)
            for ax in (axes[0, 0], axes[1, 0]):
                ax.plot(fe, mean, label=algorithm, color=colors(index), linewidth=1.3)
                ax.fill_between(fe, np.maximum(mean - std, 1e-12), mean + std, color=colors(index), alpha=0.12)
            terminal_means.append(mean[-1])
            finals.append(trajectories[:, -1])
            stats = lookup[fid, algorithm]
            times.append(stats['mean_time_clean'])
            deviations.append(stats['std_time_clean'])
            csv_write(folder / f'F{fid}_{algorithm}_curve.csv', [{'FE': int(f), 'mean': float(m), 'std': float(s)} for f, m, s in zip(fe, mean, std)], ['FE', 'mean', 'std'])
        for ax, title in ((axes[0, 0], '(a) Convergence'), (axes[1, 0], '(c) Convergence detail')):
            ax.set(xlabel='Function evaluations (FE)', ylabel='Objective value', title=title, yscale='log')
            ax.grid(alpha=0.2)
        low, high = (min(terminal_means) * 0.8, max(terminal_means) * 1.3)
        axes[1, 0].set_ylim(max(1e-12, low), max(low * 1.01, high))
        axes[0, 0].legend(fontsize=7, ncol=2)
        axes[0, 1].boxplot(finals, tick_labels=algorithms, showfliers=True)
        axes[0, 1].set(title='(b) Final objective distribution', ylabel='Objective value', yscale='log')
        axes[1, 1].bar(algorithms, times, yerr=deviations, capsize=2, color=[colors(i) for i in range(len(algorithms))])
        axes[1, 1].set(title='(d) Uninterrupted mean time (SD)', ylabel='Seconds')
        for ax in (axes[0, 1], axes[1, 1]):
            ax.tick_params(axis='x', labelrotation=70, labelsize=7)
            ax.grid(axis='y', alpha=0.2)
        included = len(algorithms)
        fig.suptitle(f'Figure {number} | F{fid} | {expected} runs/algorithm | {included}/17 complete algorithms')
        fig.tight_layout()
        fig.savefig(folder / f'Figure{number}_F{fid}.png', dpi=200)
        fig.savefig(folder / f'Figure{number}_F{fid}.pdf')
        plt.close(fig)
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', default=str(Path(__file__).resolve().parent / 'results' / 'cec17_r100'))
    parser.add_argument('--mstar-summary', default='')
    parser.add_argument('--no-plots', action='store_true')
    args = parser.parse_args()
    summarize(args.output, not args.no_plots, args.mstar_summary)
