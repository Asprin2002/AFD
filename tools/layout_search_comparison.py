"""Compare full reference scores with transparently bounded rejections."""
from math import isfinite


def compare_screening(before, after, *, baseline_us=None):
    differences, verified = [], 0
    if len(before) != len(after):
        differences.append('greedy round count')
    for number, (old, new) in enumerate(zip(before, after)):
        if ({k: v for k, v in old.items() if k != 'candidate_outcomes'} !=
                {k: v for k, v in new.items() if k != 'candidate_outcomes'}):
            differences.append(f'round {number}: selection/screening metadata')
        previous, current = old.get('candidate_outcomes', []), new.get('candidate_outcomes', [])
        if len(previous) != len(current):
            differences.append(f'round {number}: candidate count')
        for index, (prior, result) in enumerate(zip(previous, current)):
            label = f'round {number} candidate {index}'
            if prior == result:
                continue
            if result.get('status') != 'bound_pruned':
                differences.append(f'{label}: full outcome')
                continue
            if any(prior.get(key) != result.get(key) for key in ('kind', 'request_ids')):
                differences.append(f'{label}: candidate identity')
            values = [result.get(key) for key in ('gain_upper_us', 'partial_mean_us',
                                                 'incumbent_gain_us', 'tie_margin_us')]
            if not all(isinstance(value, (int, float)) and isfinite(value) for value in values):
                differences.append(f'{label}: missing/invalid bound')
                continue
            upper, partial, incumbent, margin = values
            if partial < 0 or margin < 0 or not upper < incumbent - margin:
                differences.append(f'{label}: bound does not establish a strict loss')
            if baseline_us is not None and upper < baseline_us - partial:
                differences.append(f'{label}: gain ceiling rounded inwards')
            if prior.get('status') == 'scored':
                if prior['gain_us'] > upper:
                    differences.append(f'{label}: full reference score exceeds bound')
                else:
                    verified += 1
            elif prior.get('status') not in ('service_guard', 'MemoryError', 'ValueError', 'KeyError'):
                differences.append(f'{label}: no full reference outcome to verify')
    return {'compatible': not differences, 'exact_match': before == after,
            'full_score_bounds_verified': verified, 'differences': differences}
