"""Phase 10 Pareto dominance analysis over (ndcg, catalog coverage, ild, novelty)."""

OBJECTIVES = ("ndcg", "coverage", "ild", "novelty")


def dominance(a, b):
    """True if configuration `a` dominates `b`: >= on every objective and > on at least one."""
    ge = all(a[obj] >= b[obj] for obj in OBJECTIVES)
    gt = any(a[obj] > b[obj] for obj in OBJECTIVES)
    return ge and gt


def pareto_front(configs):
    """configs: {name: {objective: value}}; returns (front_names, dominated_names, report)."""
    front = []
    dominated = []
    for name, values in configs.items():
        if any(dominance(other, values) for other_name, other in configs.items() if other_name != name):
            dominated.append(name)
        else:
            front.append(name)
    return sorted(front), sorted(dominated), {
        name: {
            "pareto_optimal": name in front,
            "dominated_by": next(
                (other_name for other_name, other in configs.items()
                 if other_name != name and dominance(other, configs[name])),
                None,
            ),
        }
        for name in configs
    }


def pick_representatives(configs, front):
    """Best accuracy / best coverage / best diversity (ILD) / best novelty among the front,
    plus a balanced pick maximizing the mean of min-max-normalized objectives."""
    if not front:
        return {}
    def best_by(obj, reverse=True):
        return max(front, key=lambda name: configs[name][obj] * (1 if reverse else -1))

    lo = {obj: min(configs[name][obj] for name in front) for obj in OBJECTIVES}
    hi = {obj: max(configs[name][obj] for name in front) for obj in OBJECTIVES}

    def balanced(name):
        vals = configs[name]
        spans = {obj: (hi[obj] - lo[obj]) or 1.0 for obj in OBJECTIVES}
        return sum((vals[obj] - lo[obj]) / spans[obj] for obj in OBJECTIVES) / len(OBJECTIVES)

    return {
        "best_accuracy": best_by("ndcg"),
        "best_catalog_coverage": best_by("coverage"),
        "best_diversity_ild": best_by("ild"),
        "best_novelty": best_by("novelty"),
        "best_balanced": max(front, key=balanced),
    }
