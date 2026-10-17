"""Measure complete searches at burst admission using lengths from a real trace.

All selected requests become decode-ready together. This is a pressure snapshot,
not a replay of the trace's original arrivals or an online latency measurement.
Run versions sequentially; --reference checks every candidate score and action.
"""
from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, replace
from hashlib import sha256
import json
from math import isfinite
from pathlib import Path
import statistics
import sys
from time import perf_counter

if __package__:
    from .layout_search_comparison import compare_screening
else:
    from layout_search_comparison import compare_screening


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--trace', required=True, type=Path)
    parser.add_argument('--start', type=int, default=256)
    parser.add_argument('--active-requests', type=int, nargs='+', default=[32, 64])
    parser.add_argument('--horizon', type=int, default=128)
    parser.add_argument('--candidates', type=int, default=12)
    parser.add_argument('--scenarios', type=int, default=4)
    parser.add_argument('--prior-mean-tokens', required=True, type=float)
    parser.add_argument('--repeat', type=int, default=3)
    parser.add_argument('--source-root', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--output-dir', required=True, type=Path)
    parser.add_argument('--reference', type=Path)
    parser.add_argument('--forecast-mode', choices=('reference', 'shared', 'bounded'))
    args = parser.parse_args()
    if (args.start < 0 or min(args.active_requests) < 1 or min(
            args.horizon, args.candidates, args.scenarios, args.repeat) < 1
            or not isfinite(args.prior_mean_tokens) or args.prior_mean_tokens <= 0):
        parser.error('counts, horizon and prior must be positive; start must be nonnegative')
    root = args.source_root.resolve()
    sys.path.insert(0, str(root))
    from elasticafd.config import load_config
    from elasticafd.layout_aware import LayoutWorld, PrefixCatalog, choose_action
    import inspect
    from elasticafd.workload import build_workload

    with args.trace.open(newline='') as handle:
        rows = list(csv.DictReader(handle))
    selected = rows[args.start:args.start + max(args.active_requests)]
    if len(selected) < max(args.active_requests):
        parser.error('trace does not contain enough requests')
    args.output_dir.mkdir(parents=True, exist_ok=False)
    metadata = {
        'arrival_semantics': 'simultaneous_decode_ready_burst; all private KV; no prefill',
        'measurement': 'one full search; no per-function timers; no search wait in device time',
        'trace': str(args.trace.resolve()), 'trace_sha256': sha256(args.trace.read_bytes()).hexdigest(),
        'start': args.start, 'active_requests': args.active_requests,
        'horizon': args.horizon, 'candidates': args.candidates, 'scenarios': args.scenarios,
        'prior_mean_tokens': args.prior_mean_tokens, 'repeat': args.repeat,
        'source_root': str(root), 'python': sys.version,
        'forecast_mode': args.forecast_mode or 'source_default',
        'configuration_sha256': sha256((root/'configs/layout-aware-online-validation.yaml').read_bytes()).hexdigest(),
        'source_sha256': {str(path.relative_to(root)): sha256(path.read_bytes()).hexdigest()
                          for path in sorted((root/'elasticafd').glob('*.py'))},
    }
    (args.output_dir/'metadata.json').write_text(json.dumps(metadata, indent=2)+'\n')
    results = []
    for count in args.active_requests:
        config = load_config(root/'configs/layout-aware-online-validation.yaml')
        requests = tuple({'request_id': row['request_id'], 'arrival_s': 0.0,
                          'input_tokens': int(row['input_tokens']),
                          'output_tokens': int(row['output_tokens'])} for row in selected[:count])
        policy = replace(config.simulation.policy, name='layout_greedy',
                         planning_horizon_tokens=args.horizon, decision_interval=args.horizon)
        config = replace(config, workload=replace(config.workload, type='closed_batch',
            requests=requests, trace_file=None, prefix_blocks=()), simulation=replace(
                config.simulation, max_batch_size=count, policy=policy,
                layout_max_candidates=args.candidates, layout_forecast_scenarios=args.scenarios,
                layout_survival_mean_tokens=args.prior_mean_tokens,
                layout_growth_reserve_tokens=args.horizon))
        for repeat in range(args.repeat):
            world = LayoutWorld(config, PrefixCatalog(()))
            for index, request in enumerate(build_workload(config.workload)):
                try:
                    world.admit(request, index % config.model.data_parallel)
                except MemoryError as exc:
                    row = {'active_requests': count, 'repeat': repeat,
                           'input_tokens': [request['input_tokens'] for request in requests],
                           'status': 'capacity_rejected', 'admitted_before_rejection': index,
                           'reason': str(exc)}
                    results.append(row)
                    with (args.output_dir/'decisions.jsonl').open('a') as handle:
                        handle.write(json.dumps(row)+'\n')
                    print(json.dumps(row), flush=True)
                    break
            else:
                row = None
            if row is not None:
                continue
            screening, reuse = [], {}
            started = perf_counter()
            mode = {'forecast_mode': args.forecast_mode} if args.forecast_mode else {}
            if mode and 'forecast_mode' not in inspect.signature(choose_action).parameters:
                parser.error('source does not support forecast-mode')
            action, gain, baseline = choose_action(world, screening=screening, reuse_statistics=reuse, **mode)
            elapsed = perf_counter() - started
            row = {'active_requests': count, 'repeat': repeat, 'wall_s': elapsed,
                   'input_tokens': [request['input_tokens'] for request in requests],
                   'action': asdict(action) if action else None, 'gain_us': gain,
                   'baseline_us': baseline, 'screening': screening, 'cost_reuse': reuse}
            results.append(row)
            with (args.output_dir/'decisions.jsonl').open('a') as handle:
                handle.write(json.dumps(row)+'\n')
            print(json.dumps({'active_requests': count, 'repeat': repeat, 'wall_s': elapsed}), flush=True)
    summary = {'cases': {}}
    for count in args.active_requests:
        values = [row['wall_s'] for row in results if row['active_requests'] == count and 'wall_s' in row]
        summary['cases'][str(count)] = {
            'completed_searches': len(values),
            'capacity_rejected': sum(row.get('status') == 'capacity_rejected' for row in results
                                     if row['active_requests'] == count),
            'min_s': min(values) if values else None,
            'median_s': statistics.median(values) if values else None,
            'max_s': max(values) if values else None}
    if args.reference:
        reference_meta = json.loads((args.reference/'metadata.json').read_text())
        differences = [f'input {key}' for key in ('arrival_semantics', 'trace_sha256', 'start',
            'horizon', 'candidates', 'scenarios', 'prior_mean_tokens',
            'configuration_sha256')
            if metadata[key] != reference_meta[key]]
        previous = [json.loads(line) for line in (args.reference/'decisions.jsonl').read_text().splitlines()]
        indexed = {(row['active_requests'], row['repeat']): row for row in previous}
        screening_exact, bounds_verified = True, 0
        # A completed sample in an interrupted larger run is still usable.
        # Only requested cases are compared; missing cases are always errors.
        for after in results:
            before = indexed.get((after['active_requests'], after['repeat']))
            if before is None:
                differences.append(f"missing reference: batch {after['active_requests']} repeat {after['repeat']}")
                continue
            for key in ('active_requests', 'repeat', 'input_tokens', 'action', 'gain_us',
                        'baseline_us', 'status', 'reason', 'admitted_before_rejection'):
                if before.get(key) != json.loads(json.dumps(after.get(key))):
                    differences.append(f"batch {after['active_requests']} repeat {after['repeat']}: {key}")
            check = compare_screening(before.get('screening', []), json.loads(json.dumps(after.get('screening', []))),
                                      baseline_us=after.get('baseline_us'))
            differences.extend(check['differences'])
            screening_exact &= check['exact_match']
            bounds_verified += check['full_score_bounds_verified']
        summary['comparison'] = {'exact_match': not differences and screening_exact, 'differences': differences,
                                 'compared_searches': len(results), 'reference_searches': len(previous)}
        if not screening_exact:
            summary['comparison'].update(decision_output_match=not differences,
                                         full_score_bounds_verified=bounds_verified)
    (args.output_dir/'timings.json').write_text(json.dumps(summary, indent=2)+'\n')
    print(json.dumps(summary), flush=True)
    if args.reference and summary['comparison']['differences']:
        raise SystemExit('reference comparison failed')


if __name__ == '__main__':
    main()
