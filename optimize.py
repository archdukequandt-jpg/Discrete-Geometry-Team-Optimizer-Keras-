import numpy as np
import random
from itertools import combinations

def avg_pair_score(indices, pair_score_fn):
    if len(indices) < 2:
        return 0.0
    s = 0.0; n = 0
    for i,j in combinations(indices, 2):
        s += pair_score_fn(i,j)
        n += 1
    return s / max(1,n)

def objective(groups, pair_score_fn=None, group_score_fn=None, var_weight=0.25):
    """Compute optimization objective.

    If group_score_fn is provided, it is used directly to score each group
    (higher is better). Otherwise, the score for each group is the average
    pairwise score using pair_score_fn.
    """
    if group_score_fn is not None:
        scores = [float(group_score_fn(g)) for g in groups]
    else:
        if pair_score_fn is None:
            raise ValueError("pair_score_fn is required when group_score_fn is None")
        scores = [avg_pair_score(g, pair_score_fn) for g in groups]
    mean_s = float(np.mean(scores)) if scores else 0.0
    var_s = float(np.var(scores)) if scores else 0.0
    return mean_s - var_weight * var_s, {"group_scores": scores, "mean": mean_s, "var": var_s}

def random_groups(n_items, group_size, n_groups, rng=None):
    rng = rng or random.Random(42)
    all_idx = list(range(n_items))
    rng.shuffle(all_idx)
    needed = group_size * n_groups
    if needed > n_items:
        raise ValueError("Not enough items for requested grouping.")
    all_idx = all_idx[:needed]
    groups = [all_idx[i*group_size:(i+1)*group_size] for i in range(n_groups)]
    return groups

def hillclimb(groups, pair_score_fn=None, group_score_fn=None, n_items=None, iters=2000, var_weight=0.25, rng=None):
    rng = rng or random.Random(42)
    best = [g[:] for g in groups]
    if n_items is None:
        raise ValueError("n_items is required")
    best_score, best_meta = objective(best, pair_score_fn=pair_score_fn, group_score_fn=group_score_fn, var_weight=var_weight)
    group_size = len(best[0]) if best else 0
    used = set([i for g in best for i in g])

    for t in range(iters):
        cand = [g[:] for g in best]
        # choose operation: swap within used set, or swap with unused
        if rng.random() < 0.7 and len(used) < n_items:
            # swap one element with unused
            gk = rng.randrange(len(cand))
            pos = rng.randrange(group_size)
            old = cand[gk][pos]
            # pick unused
            unused = [i for i in range(n_items) if i not in used]
            new = rng.choice(unused)
            cand[gk][pos] = new
        else:
            # swap between two groups
            g1, g2 = rng.sample(range(len(cand)), 2)
            p1 = rng.randrange(group_size); p2 = rng.randrange(group_size)
            cand[g1][p1], cand[g2][p2] = cand[g2][p2], cand[g1][p1]

        # ensure uniqueness within groups
        ok = True
        for g in cand:
            if len(set(g)) != len(g):
                ok = False; break
        if not ok:
            continue

        score, meta = objective(cand, pair_score_fn=pair_score_fn, group_score_fn=group_score_fn, var_weight=var_weight)
        if score > best_score:
            best = cand
            best_score = score
            best_meta = meta
            used = set([i for g in best for i in g])

    return best, best_score, best_meta
