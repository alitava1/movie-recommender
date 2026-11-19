"""Phase 13 experiment tracks A-E, preregistered statistics, and artifact assembly."""

import time

from .. import evaluation
from ..phase11.statistics import holm_bonferroni, paired_bootstrap_with_p
from ..phase7.cf_adapter import build_cf
from . import engine
from .data import TRAITS, full_audit
from .profiles import (
    ablate_single_trait,
    make_profile,
    missing_profiles,
    noisy_profiles,
    normalize_traits,
    randomized_profiles,
    shuffled_profiles,
)

SEED = 2026


class StratumModels:
    """Everything derived from a stratum's permitted-training rows only (leak-safe)."""

    ATTR_NAMES = ("complexity", "novelty", "emotional_intensity", "social_orientation", "predictability", "release_norm")

    def __init__(self, train_rows, traits_raw, name, with_cf=True, power_iters=10):
        import gc

        self._traits_raw = traits_raw
        self.name = name
        self.train_rows = train_rows
        self.pop_counts = self._pop_counts(train_rows)
        self.catalog = sorted({mid for _, mid, _, _ in train_rows})
        self.item_index = {mid: i for i, mid in enumerate(self.catalog)}
        self.attr, _ = engine.build_attr_matrix(train_rows, self.catalog, self.item_index, power_iters=power_iters)
        self.attr_get = self._attr_closure()
        if with_cf:
            self.cf_model = build_cf(train_rows, neighbors=100, shrinkage=10.0, prediction_shrinkage=0.0)
        else:
            self.cf_model = None
        self.pred_shrinkage = 300.0
        self.hist_by_user = {}
        for uid, mid, _, _ in train_rows:
            self.hist_by_user.setdefault(uid, []).append(mid)
        ref_users = sorted({uid for uid, _, _, _ in train_rows})
        self.traits_norm, self.trait_bounds = normalize_traits(traits_raw, ref_users)
        self.weights = engine.fit_ridge(train_rows, self.traits_norm, self.item_index, self.attr)
        self.model = engine.ridge_model(self.weights)
        self.bitsets = engine.build_item_bitsets(train_rows)
        self.pop_pool = engine.pop_pool_fn(self.pop_counts, self.hist_by_user, self.catalog)
        self._base_cache = {}
        gc.collect()

    def _pop_counts(self, train_rows):
        counts = {}
        for _, mid, _, _ in train_rows:
            counts[mid] = counts.get(mid, 0) + 1
        return counts

    def _attr_closure(self):
        cache = {}
        names = self.ATTR_NAMES

        def get(mid):
            if mid not in cache:
                i = self.item_index.get(mid)
                cache[mid] = (
                    {name: float(self.attr[i, j]) for j, name in enumerate(names)}
                    if i is not None else {name: 0.5 for name in names}
                )
            return cache[mid]

        return get

    def base_scores(self, uid):
        """Phase 10-style hybrid CF+popularity base over the fixed popularity@200 pool (cached)."""
        if uid in self._base_cache:
            return self._base_cache[uid]
        pool = self.pop_pool(uid)
        ur = {mid: 4.0 for mid in self.hist_by_user.get(uid, [])}
        scores, cf_n = engine.hybrid_base_scores(pool, ur, self.cf_model, self.pop_counts, self.pred_shrinkage)
        self._base_cache[uid] = (pool, scores, cf_n)
        return pool, scores, cf_n

    def evaluate_conditions(self, targets, conditions, k=10):
        """conditions: {name: (traits_norm_or_None, weight, enabled)} over fixed pools."""
        per_user = {name: {} for name in conditions}
        rec_lists = {name: {} for name in conditions}
        t0 = time.perf_counter()
        for uid in sorted(targets):
            pool, base, _ = self.base_scores(uid)
            for name, (traits, weight, enabled) in conditions.items():
                tn = traits.get(uid) if traits is not None else None
                recs, _s, _d = engine.rank_with_personality13(
                    pool, base, tn, uid, self.model, self.attr_get, weight, enabled, k=k,
                )
                per_user[name][f"user:{uid}"] = engine.evaluate(recs, targets[uid], self.pop_counts, self.bitsets, k=k)
                rec_lists[name][f"user:{uid}"] = recs
        summary = {
            name: engine.summarize(per_user[name], rec_lists[name].values(), len(self.catalog), self.pop_counts)
            for name in conditions
        }
        return {"per_user": per_user, "summary": summary, "rec_lists": rec_lists,
                "wall_seconds": time.perf_counter() - t0}

    def select_weight(self):
        """Inner-validation w_p selection on a chronological 80/20 split of permitted history."""
        by_user = engine.per_user_ordered(self.train_rows)
        inner_train, inner_targets = [], {}
        for uid, items in by_user.items():
            cut = max(1, int(len(items) * 0.8))
            later = [(mid, rating, ts) for mid, rating, ts in items[cut:] if rating >= engine.RELEVANCE]
            if not later:
                continue
            inner_targets[uid] = {mid for mid, _, _ in later}
            inner_train.extend((uid, mid, rating, ts) for mid, rating, ts in items[:cut])
        if len(inner_targets) < 50:
            return 0.15, {"fallback": "insufficient inner-validation targets; diagnostic weight fixed"}
        inner = StratumModels(inner_train, self._traits_raw, self.name + "_inner")
        sweep = []
        for wp in engine.WEIGHT_GRID:
            res = inner.evaluate_conditions(inner_targets, {"cand": (None, wp, wp > 0.0)})
            sweep.append({"weight": wp, "ndcg": round(res["summary"]["cand"]["ndcg"], 6)})
        best = max(sweep, key=lambda row: (row["ndcg"], -row["weight"]))
        return best["weight"], {"sweep": sweep, "inner_targets": len(inner_targets)}


def run_baseline(traits_raw, rows, smoke_users=None):
    """Track B: warm-stratum baselines + candidate recall diagnostics."""
    train, targets, excluded = engine.warm_stratum(rows)
    if smoke_users is not None:
        train = [r for r in train if r[0] in smoke_users]
        targets = {u: t for u, t in targets.items() if u in smoke_users}
    models = StratumModels(train, traits_raw, "warm")
    recall = engine.pool_recall(models.pop_pool, targets, m_values=(100, 200))
    per_user = {"popularity_only": {}, "cf_only": {}, "personality_disabled": {}}
    rec_lists = {name: {} for name in per_user}
    for uid in sorted(targets):
        pool, base, cf_n = models.base_scores(uid)
        for name, recs in (
            ("popularity_only", [mid for mid, _ in sorted(((m, models.pop_counts.get(m, 0)) for m in pool), key=lambda r: (-r[1], r[0]))][:10]),
            ("personality_disabled", sorted(base, key=lambda m: (-base[m], m))[:10]),
        ):
            per_user[name][f"user:{uid}"] = engine.evaluate(recs, targets[uid], models.pop_counts, models.bitsets)
            rec_lists[name][f"user:{uid}"] = recs
        ur = {mid: 4.0 for mid in models.hist_by_user.get(uid, [])}
        evidence = models.cf_model.evidence(ur, set(pool)) if ur else {}
        cf_scores = {mid: min(5.0, max(0.5, mean + num / (den + models.pred_shrinkage)))
                     for mid, (mean, num, den) in evidence.items()}
        recs = [mid for mid, _ in sorted(cf_scores.items(), key=lambda r: (-r[1], r[0]))][:10]
        per_user["cf_only"][f"user:{uid}"] = engine.evaluate(recs, targets[uid], models.pop_counts, models.bitsets)
        rec_lists["cf_only"][f"user:{uid}"] = recs
    summary = {name: engine.summarize(per_user[name], rec_lists[name].values(), len(models.catalog), models.pop_counts)
               for name in per_user}
    return {
        "train_rows": len(train), "targets_count": len(targets), "excluded": len(excluded),
        "catalog_size": len(models.catalog),
        "candidate_recall": recall,
        "summary": summary, "per_user": per_user, "rec_lists": rec_lists,
        "targets": targets, "models": models,
    }


def run_warm_personality(baseline, smoke_users=None):
    """Track C: P1-P3 conditions on the warm stratum with inner-selected w_p."""
    models = baseline["models"]
    targets = {u: t for u, t in baseline["targets"].items()}
    if smoke_users is not None:
        targets = {u: t for u, t in targets.items() if u in smoke_users}
    w_sel, sel_info = models.select_weight()
    controls = {
        "shuffled": shuffled_profiles(models.traits_norm, seed=SEED + 1),
        "random": randomized_profiles(models.traits_norm, seed=SEED + 2),
        "noisy": noisy_profiles(models.traits_norm, sigma=0.7, seed=SEED + 3),
        "missing": missing_profiles(models.traits_norm, n_missing=2, seed=SEED + 4),
    }
    conditions = {
        "personality_disabled": (None, 0.0, False),
        "personality_selected": (models.traits_norm, w_sel, w_sel > 0.0),
        "correct_w15": (models.traits_norm, 0.15, True),
        "shuffled_w15": (controls["shuffled"], 0.15, True),
        "random_w15": (controls["random"], 0.15, True),
        "noisy_w15": (controls["noisy"], 0.15, True),
        "missing_w15": (controls["missing"], 0.15, True),
    }
    res = models.evaluate_conditions(targets, conditions)
    res["inner_selected_weight"] = w_sel
    res["selection_info"] = sel_info
    res["targets_count"] = len(targets)
    return res


def run_sparse(traits_raw, rows, smoke_users=None):
    """Track D: K in {1,5,10,20} simulated sparse histories with mandatory controls."""
    out = {}
    for k in engine.K_VALUES:
        train, targets, excluded = engine.sparse_stratum(rows, k)
        if smoke_users is not None:
            targets = {u: t for u, t in targets.items() if u in smoke_users}
            train = [r for r in train if r[0] in smoke_users]
        models = StratumModels(train, traits_raw, f"sparse_k{k}")
        w_sel, sel_info = models.select_weight()
        controls = {
            "shuffled": shuffled_profiles(models.traits_norm, seed=SEED + 10 + k),
            "random": randomized_profiles(models.traits_norm, seed=SEED + 20 + k),
            "noisy": noisy_profiles(models.traits_norm, sigma=0.7, seed=SEED + 30 + k),
        }
        conditions = {
            "baseline_disabled": (None, 0.0, False),
            "correct_selected": (models.traits_norm, w_sel, w_sel > 0.0),
            "correct_w15": (models.traits_norm, 0.15, True),
            "shuffled_w15": (controls["shuffled"], 0.15, True),
            "random_w15": (controls["random"], 0.15, True),
            "noisy_w15": (controls["noisy"], 0.15, True),
        }
        res = models.evaluate_conditions(targets, conditions)
        res.update({"k": k, "targets": len(targets), "excluded": len(excluded),
                    "inner_selected_weight": w_sel, "selection_info": sel_info})
        out[f"K{k}"] = res
    return out


def run_zero_history(traits_raw, rows, smoke_users=None):
    """Track E: 10-fold participant-disjoint leave-one-user-out zero-history evaluation."""
    by_user = engine.per_user_ordered(rows)
    all_users = sorted(by_user)
    if smoke_users is not None:
        all_users = [u for u in all_users if u in smoke_users]
    folds = engine.participant_folds(len(all_users), n_folds=10)
    per_user = {"personality_only": {}, "popularity_only": {}, "personality_popularity": {}, "random_profile": {}}
    rec_lists = {name: {} for name in per_user}
    selected_weights = []
    for f_idx, fold_users in enumerate(folds):
        fold_set = set(fold_users)
        train_rows = [r for r in rows if r[0] not in fold_set]
        # Zero-history conditions never use CF (empty user history); skip the CF build entirely.
        models = StratumModels(train_rows, traits_raw, f"zero_fold{f_idx}", with_cf=False, power_iters=5)
        rel = {}
        for uid in fold_users:
            rel[uid] = {mid for mid, rating, _ in by_user[uid] if rating >= engine.RELEVANCE}
        rel = {u: r for u, r in rel.items() if r}
        # Blend weight selected on ONE inner fold (first 80% of training participants) to avoid
        # per-fold tuning cost; identical for all folds (documented).
        if not selected_weights:
            inner_pool_users = sorted(models.hist_by_user)[: int(len(models.hist_by_user) * 0.8)]
            inner_targets = {}
            for uid in inner_pool_users:
                elig = {mid for mid, rating, _ in by_user.get(uid, []) if rating >= engine.RELEVANCE}
                if elig:
                    inner_targets[uid] = elig
            if len(inner_targets) >= 50:
                sweep = []
                for wp in engine.WEIGHT_GRID:
                    acc, n = 0.0, 0
                    for uid, relset in list(inner_targets.items())[:300]:
                        pool = models.pop_pool(uid)
                        ur = {mid: 4.0 for mid in models.hist_by_user.get(uid, [])}
                        base, _ = engine.hybrid_base_scores(pool, ur, models.cf_model, models.pop_counts, models.pred_shrinkage)
                        recs, _, _ = engine.rank_with_personality13(
                            pool, base, models.traits_norm.get(uid), uid, models.model, models.attr_get, wp, wp > 0.0, k=10)
                        acc += evaluation.ndcg_at_k(recs, relset, 10)
                        n += 1
                    sweep.append({"weight": wp, "ndcg": acc / max(1, n)})
                best = max(sweep, key=lambda r: (r["ndcg"], -r["weight"]))
                selected_weights.append(best["weight"])
            else:
                selected_weights.append(0.15)
        w_sel = selected_weights[0]
        random_traits = randomized_profiles(models.traits_norm, seed=SEED + 100 + f_idx)
        for uid in sorted(rel):
            pool = models.pop_pool(uid)
            pop_rec = [mid for mid, _ in sorted(((m, models.pop_counts.get(m, 0)) for m in pool), key=lambda r: (-r[1], r[0]))][:10]
            pers = engine.ridge_scores(models.model, make_profile(uid, models.traits_norm[uid]), models.attr_get, pool)
            pers_rec = [mid for mid, _ in sorted(pers.items(), key=lambda r: (-r[1], r[0]))][:10]
            rnd = engine.ridge_scores(models.model, make_profile(uid, random_traits[uid]), models.attr_get, pool)
            rnd_rec = [mid for mid, _ in sorted(rnd.items(), key=lambda r: (-r[1], r[0]))][:10]
            ur = {}
            base, _ = engine.hybrid_base_scores(pool, ur, None, models.pop_counts, models.pred_shrinkage)
            blend_rec, _, _ = engine.rank_with_personality13(
                pool, base, models.traits_norm[uid], uid, models.model, models.attr_get, w_sel, w_sel > 0.0, k=10)
            for name, recs in (("personality_only", pers_rec), ("popularity_only", pop_rec),
                               ("personality_popularity", blend_rec), ("random_profile", rnd_rec)):
                per_user[name][f"f{f_idx}:user:{uid}"] = engine.evaluate(recs, rel[uid], models.pop_counts, models.bitsets)
                rec_lists[name][f"f{f_idx}:user:{uid}"] = recs
    summary = {name: engine.summarize(per_user[name], rec_lists[name].values(),
                                      len(set(mid for _, mid, _, _ in rows)), _pop_all(rows))
               for name in per_user}
    return {"summary": summary, "per_user": per_user, "blend_weight": selected_weights[0] if selected_weights else 0.15}


def _pop_all(rows):
    counts = {}
    for _, mid, _, _ in rows:
        counts[mid] = counts.get(mid, 0) + 1
    return counts


def run_statistics(baseline, warm, sparse, zero, resamples=2000):
    """The five preregistered primary comparisons (Holm-Bonferroni) + exploratory rows."""
    raw = []

    def add(name, family, left, right, metric="ndcg"):
        comp = paired_bootstrap_with_p(left, right, metric=metric, resamples=resamples, confidence=0.95, seed=SEED)
        raw.append({
            "comparison": name, "family": family, "metric": metric,
            "abs_diff": comp["abs_diff"], "rel_diff": comp["rel_diff"],
            "ci_lower": comp["abs_lower"], "ci_upper": comp["abs_upper"],
            "includes_zero": comp["includes_zero"], "p_value": comp["p_value"],
            "users": comp["users"], "observations": comp["observations"],
        })

    add("P1_personality_vs_baseline_warm", "primary", warm["per_user"]["personality_selected"], baseline["per_user"]["personality_disabled"])
    add("P2_correct_vs_shuffled_warm", "primary", warm["per_user"]["correct_w15"], warm["per_user"]["shuffled_w15"])
    add("P3_correct_vs_random_warm", "primary", warm["per_user"]["correct_w15"], warm["per_user"]["random_w15"])
    add("P4_personality_only_vs_popularity_zero", "primary", zero["per_user"]["personality_only"], zero["per_user"]["popularity_only"])
    add("P5_personality_vs_sparse_baseline_K5", "primary", sparse["K5"]["per_user"]["correct_selected"], sparse["K5"]["per_user"]["baseline_disabled"])
    # exploratory
    add("E_correct_vs_noisy_warm", "exploratory", warm["per_user"]["correct_w15"], warm["per_user"]["noisy_w15"])
    add("E_correct_vs_missing_warm", "exploratory", warm["per_user"]["correct_w15"], warm["per_user"]["missing_w15"])
    for k in ("K1", "K10", "K20"):
        add(f"E_sparse_{k}_correct_vs_baseline", "exploratory", sparse[k]["per_user"]["correct_selected"], sparse[k]["per_user"]["baseline_disabled"])
        add(f"E_sparse_{k}_correct_vs_shuffled", "exploratory", sparse[k]["per_user"]["correct_w15"], sparse[k]["per_user"]["shuffled_w15"])
    add("E_zero_personality_pop_blend_vs_popularity", "exploratory", zero["per_user"]["personality_popularity"], zero["per_user"]["popularity_only"])
    add("E_zero_correct_vs_random_profile", "exploratory", zero["per_user"]["personality_only"], zero["per_user"]["random_profile"])
    add("E_cf_only_vs_hybrid_warm", "exploratory", baseline["per_user"]["cf_only"], baseline["per_user"]["personality_disabled"])
    return holm_bonferroni(raw)


def run_ablations(models, targets, traits_raw):
    """Score-level sensitivity: noise levels, missing dimensions, per-trait ablation (w=0.15)."""
    rows = []
    for sigma in (0.3, 0.7, 1.2):
        noisy = noisy_profiles(models.traits_norm, sigma=sigma, seed=SEED + 200 + int(sigma * 10))
        res = models.evaluate_conditions(targets, {f"noise_{sigma}": (noisy, 0.15, True)})
        rows.append({"ablation": "noise_sigma", "level": sigma,
                     "ndcg": round(res["summary"][f"noise_{sigma}"]["ndcg"], 6),
                     "hr": round(res["summary"][f"noise_{sigma}"]["hr"], 6),
                     "coverage": round(res["summary"][f"noise_{sigma}"]["coverage"], 6)})
    for n_miss in (1, 2, 3):
        masked = missing_profiles(models.traits_norm, n_miss, seed=SEED + 300 + n_miss)
        res = models.evaluate_conditions(targets, {f"missing_{n_miss}": (masked, 0.15, True)})
        rows.append({"ablation": "missing_dims", "level": n_miss,
                     "ndcg": round(res["summary"][f"missing_{n_miss}"]["ndcg"], 6),
                     "hr": round(res["summary"][f"missing_{n_miss}"]["hr"], 6),
                     "coverage": round(res["summary"][f"missing_{n_miss}"]["coverage"], 6)})
    for trait in TRAITS:
        abl = ablate_single_trait(models.traits_norm, trait)
        res = models.evaluate_conditions(targets, {f"ablate_{trait}": (abl, 0.15, True)})
        rows.append({"ablation": "trait_ablation", "level": trait,
                     "ndcg": round(res["summary"][f"ablate_{trait}"]["ndcg"], 6),
                     "hr": round(res["summary"][f"ablate_{trait}"]["hr"], 6),
                     "coverage": round(res["summary"][f"ablate_{trait}"]["coverage"], 6)})
    return rows


def segment_by_history(models, per_user, targets):
    """Warm users grouped by permitted-history size terciles (training-only cut points)."""
    sizes = {uid: len(items) for uid, items in engine.per_user_ordered(models.train_rows).items()}
    vals = sorted(sizes.values())
    q1, q2 = vals[len(vals) // 3], vals[2 * len(vals) // 3]
    segments = {"small": [], "medium": [], "large": []}
    for uid in targets:
        s = sizes.get(uid, 0)
        segments["small" if s <= q1 else "medium" if s <= q2 else "large"].append(f"user:{uid}")
    out = {}
    for name, keys in segments.items():
        rows = [per_user[k] for k in keys if k in per_user]
        out[name] = {"users": len(rows), "cutpoints": [q1, q2]}
        if rows:
            out[name].update({m: sum(r[m] for r in rows) / len(rows)
                              for m in ("ndcg", "map", "hr", "ild", "novelty")})
    return out
