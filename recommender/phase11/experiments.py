"""Phase 11 Track A (synthetic benchmark) and Track B (ML-1M architectural integration) runners."""

import time

from .. import content as content_mod
from ..data import group_by_user
from ..personality.evaluation import list_change_rate, trait_recovery_diagnostics
from ..personality.preference_model import RidgePersonalityModel
from ..personality.profile import generate_population_profiles
from ..personality.questionnaire import build_schema
from ..personality.simulator import SCENARIOS, build_movie_attributes, generate_synthetic_dataset
from ..phase7.metrics import evaluate_user_phase7, summarize_users
from ..phase7.temporal import split_inner_chronological
from ..phase9.fusion import merge_candidates
from ..phase9.generators import CFCandidateGenerator, CandidateList, PopularityCandidateGenerator
from ..phase10.diversity import rank_candidates
from ..phase10.experiments import segment_analysis
from ..phase10.metrics import novelty_stats, popularity_concentration, unique_genre_representation
from ..phase10.pareto import pareto_front, pick_representatives
from ..phase10.scoring import extract_components, normalize_components
from .personality import build_control_profiles, mask_dimensions
from .ranking import rank_with_personality

WEIGHT_GRID = (0.0, 0.05, 0.10, 0.15, 0.20, 0.30)
DIAGNOSTIC_ACTIVE_WEIGHT = 0.15
TRACK_A_SCENARIOS = ("A_null", "B_weak", "C_moderate", "D_strong", "E_confounded")
TRACK_A_SEEDS = (101, 202, 303)


def _prepare_three_w3_pools(train_ratings, target_relevant, movies, vectors, settings, pool_m=200):
    """Build fixed Phase 10 cf_popularity@200 pools and three_w3 base scores once per user."""
    cf_gen = CFCandidateGenerator(
        train_ratings,
        neighbors=settings.get("cf_tuning", {}).get("neighbors", 100),
        shrinkage=settings["cf"]["shrinkage"],
        prediction_shrinkage=settings.get("cf_tuning", {}).get("prediction_shrinkage", 300.0),
    )
    pop_gen = PopularityCandidateGenerator(train_ratings)
    pop_counts = dict(pop_gen.positive_counts)
    train_users = group_by_user(train_ratings)
    pred_shrinkage = settings.get("cf_tuning", {}).get("prediction_shrinkage", 300.0)

    user_pools = {}
    for uid in sorted(target_relevant):
        ur = train_users.get(uid, {})
        if not ur:
            # Cold user: three_w3 reduces to training popularity over unseen catalog.
            pop_list = pop_gen.generate({}, movies, pool_m)
            pool_ids = pop_list.ids()
            components, _ = extract_components({}, pool_ids, {}, pop_counts, {}, movies)
            norm = normalize_components(components)
            _, base_scores, _ = rank_candidates(pool_ids, norm, "popularity_only")
            user_pools[uid] = (pool_ids, base_scores)
            continue

        evidence = cf_gen.model.evidence(ur, movies.keys() - ur.keys())
        cf_scores = {
            mid: min(5.0, max(0.5, mean + num / (den + pred_shrinkage)))
            for mid, (mean, num, den) in evidence.items()
        }
        cf_list = CandidateList("cf", tuple(sorted(cf_scores.items(), key=lambda r: (-r[1], r[0]))[:pool_m]))
        pop_list = pop_gen.generate(ur, movies, pool_m)
        pool_ids = merge_candidates([cf_list, pop_list], pool_m, mode="union", quotas={"cf": 0.5, "popularity": 0.5}).ids()
        prof = content_mod.user_profile(vectors, ur)
        cont_scores = content_mod.score(prof, vectors, set(pool_ids)) if prof else {}
        components, _ = extract_components(ur, pool_ids, cf_scores, pop_counts, cont_scores, movies)
        norm = normalize_components(components)
        _, base_scores, _ = rank_candidates(
            pool_ids, norm, "three_signal", {"weights": {"cf": 0.4, "popularity": 0.2, "content": 0.4}},
        )
        user_pools[uid] = (pool_ids, base_scores)
    return user_pools, pop_counts


def _select_weight_on_inner(user_pools, relevant_by_user, profiles, movie_features, pers_model, vectors, pop_counts, weight_grid=WEIGHT_GRID):
    """Select personality_weight strictly on inner-validation NDCG@10."""
    sweep = []
    for wp in weight_grid:
        ndcg_sum = 0.0
        n = 0
        for uid, (pool_ids, base_scores) in user_pools.items():
            recs, _, _ = rank_with_personality(
                pool_ids, base_scores, profiles.get(uid), movie_features, pers_model,
                personality_weight=wp, enabled=(wp > 0.0), k=10,
            )
            m = evaluate_user_phase7(recs, relevant_by_user[uid], 10, vectors, pop_counts)
            ndcg_sum += m["ndcg"]
            n += 1
        sweep.append({"weight": wp, "ndcg": ndcg_sum / max(1, n)})
    best = max(sweep, key=lambda row: (row["ndcg"], -row["weight"]))
    return best["weight"], sweep


def _score_conditions(
    user_pools,
    relevant_by_user,
    conditions,
    movie_features,
    pers_model,
    movies,
    vectors,
    pop_counts,
    key_prefix="",
):
    """Evaluate a dict of {cond_name: (profiles_or_None, weight, enabled, only_pers_catalog)} over fixed pools."""
    per_user = {name: {} for name in conditions}
    cov_sets = {name: set() for name in conditions}
    ranked_lists = {name: {} for name in conditions}
    all_mids = tuple(sorted(movies))

    for uid in sorted(user_pools):
        pool_ids, base_scores = user_pools[uid]
        rel = relevant_by_user[uid]
        key = f"{key_prefix}{uid}" if key_prefix else f"user:{uid}"
        for name, (profs, wp, enabled, pers_only) in conditions.items():
            prof = profs.get(uid) if profs is not None else None
            if pers_only:
                if prof is None:
                    recs = []
                else:
                    scores = pers_model.score_candidates(prof, movie_features, all_mids)
                    recs = [mid for mid, _ in sorted(scores.items(), key=lambda r: (-r[1], r[0]))][:10]
            else:
                recs, _, _ = rank_with_personality(
                    pool_ids, base_scores, prof, movie_features, pers_model,
                    personality_weight=wp, enabled=enabled, k=10,
                )
            per_user[name][key] = evaluate_user_phase7(recs, rel, 10, vectors, pop_counts)
            cov_sets[name].update(recs)
            ranked_lists[name][key] = recs

    summary = {}
    for name in conditions:
        s = summarize_users(per_user[name], len(cov_sets[name]), len(movies))
        lists = list(ranked_lists[name].values())
        nov = novelty_stats(lists, pop_counts)
        s["novelty_p11"] = nov["novelty"]
        s["long_tail_share"] = nov["long_tail_share"]
        s["popularity_concentration"] = popularity_concentration(lists, pop_counts)
        s["unique_genres"] = unique_genre_representation(lists, movies)
        summary[name] = s
    return per_user, summary, cov_sets, ranked_lists


def _relevant_map(rows, threshold=4.0):
    rel = {}
    for uid, mid, rating, _ in rows:
        if rating >= threshold:
            rel.setdefault(uid, set()).add(mid)
    return rel


def run_track_a(movies, vectors, settings, scenarios=TRACK_A_SCENARIOS, seeds=TRACK_A_SEEDS, n_users=360, interactions_per_user=50):
    """Track A: Phase 8 controlled synthetic benchmark scored with Phase 10 three_w3 + RidgePersonality."""
    schema = build_schema()
    t0 = time.perf_counter()
    scenario_out = {}

    for scenario in scenarios:
        warm_per_user = {}
        cold_per_user = {}
        warm_cov = {}
        cold_cov = {}
        warm_lists = {}
        cold_lists = {}
        selected_weights = []
        pop_counts_ref = {}
        recovery_stats = []

        for seed in seeds:
            ds = generate_synthetic_dataset(
                movies, scenario, seed, n_users=n_users, interactions_per_user=interactions_per_user, schema=schema,
            )
            recovery_stats.append(trait_recovery_diagnostics(ds._latent_traits, ds.observed_profiles)["mean_pearson_r"])
            controls = build_control_profiles(
                ds.observed_profiles.keys(), schema, seed=seed + 5000,
                latent_override=ds._latent_traits, base_profiles=ds.observed_profiles,
            )
            # Inner validation on warm_train -> warm_val (zero test leakage).
            pers_inner = RidgePersonalityModel(alpha=5.0).fit(ds.warm_train, controls["correct"], ds.movie_features)
            rel_val = _relevant_map(ds.warm_val)
            val_pools, val_pop = _prepare_three_w3_pools(ds.warm_train, rel_val, movies, vectors, settings)
            sel_w, _sweep = _select_weight_on_inner(
                val_pools, rel_val, controls["correct"], ds.movie_features, pers_inner, vectors, val_pop,
            )
            selected_weights.append(sel_w)
            active_w = sel_w if sel_w > 0.0 else DIAGNOSTIC_ACTIVE_WEIGHT

            # Refit on warm_train + warm_val for outer warm & cold evaluation.
            warm_fit_rows = ds.warm_train + ds.warm_val
            pers_outer = RidgePersonalityModel(alpha=5.0).fit(warm_fit_rows, controls["correct"], ds.movie_features)

            rel_warm_test = _relevant_map(ds.warm_test)
            warm_pools, warm_pop = _prepare_three_w3_pools(warm_fit_rows, rel_warm_test, movies, vectors, settings)
            pop_counts_ref = warm_pop
            warm_conds = {
                "three_w3_baseline": (None, 0.0, False, False),
                "personality_disabled": (controls["correct"], sel_w, False, False),
                "personality_selected": (controls["correct"], sel_w, True, False),
                "correct_active": (controls["correct"], active_w, True, False),
                "noisy_active": (controls["noisy"], active_w, True, False),
                "missing_active": (controls["missing"], active_w, True, False),
                "shuffled_active": (controls["shuffled"], active_w, True, False),
                "random_active": (controls["random"], active_w, True, False),
                "personality_only": (controls["correct"], 1.0, True, True),
            }
            pu_w, _, cov_w, lst_w = _score_conditions(
                warm_pools, rel_warm_test, warm_conds, ds.movie_features, pers_outer, movies, vectors, warm_pop,
                key_prefix=f"s{seed}:",
            )
            for cname in warm_conds:
                warm_per_user.setdefault(cname, {}).update(pu_w[cname])
                warm_cov.setdefault(cname, set()).update(cov_w[cname])
                warm_lists.setdefault(cname, []).extend(lst_w[cname].values())

            rel_cold = _relevant_map(ds.cold_test_ratings)
            cold_pools, _ = _prepare_three_w3_pools(warm_fit_rows, rel_cold, movies, vectors, settings)
            cold_conds = {
                "cold_baseline_pop": (None, 0.0, False, False),
                "cold_personality_selected": (controls["correct"], sel_w, True, False),
                "cold_correct_active": (controls["correct"], active_w, True, False),
                "cold_noisy_active": (controls["noisy"], active_w, True, False),
                "cold_missing_active": (controls["missing"], active_w, True, False),
                "cold_shuffled_active": (controls["shuffled"], active_w, True, False),
                "cold_random_active": (controls["random"], active_w, True, False),
                "cold_personality_only": (controls["correct"], 1.0, True, True),
            }
            pu_c, _, cov_c, lst_c = _score_conditions(
                cold_pools, rel_cold, cold_conds, ds.movie_features, pers_outer, movies, vectors, warm_pop,
                key_prefix=f"s{seed}:",
            )
            for cname in cold_conds:
                cold_per_user.setdefault(cname, {}).update(pu_c[cname])
                cold_cov.setdefault(cname, set()).update(cov_c[cname])
                cold_lists.setdefault(cname, []).extend(lst_c[cname].values())

        def _summarize_pooled(pu_map, cov_map, lst_map):
            out = {}
            for cname in pu_map:
                s = summarize_users(pu_map[cname], len(cov_map[cname]), len(movies))
                nov = novelty_stats(lst_map[cname], pop_counts_ref)
                s["novelty_p11"] = nov["novelty"]
                s["long_tail_share"] = nov["long_tail_share"]
                s["popularity_concentration"] = popularity_concentration(lst_map[cname], pop_counts_ref)
                s["unique_genres"] = unique_genre_representation(lst_map[cname], movies)
                out[cname] = s
            return out

        scenario_out[scenario] = {
            "selected_weights": selected_weights,
            "mean_selected_weight": sum(selected_weights) / len(selected_weights),
            "mean_trait_recovery_r": sum(recovery_stats) / len(recovery_stats),
            "warm_summary": _summarize_pooled(warm_per_user, warm_cov, warm_lists),
            "warm_per_user": warm_per_user,
            "cold_summary": _summarize_pooled(cold_per_user, cold_cov, cold_lists),
            "cold_per_user": cold_per_user,
        }

    # Noise and missing-dimension ablations on Scenario C_moderate (pooled across seeds).
    ablations = {"noise": [], "missing_dims": []}
    for sigma in (0.20, 0.45, 0.80, 1.20):
        pu_acc = {}
        cov_acc = set()
        rs, maes = [], []
        for seed in seeds:
            ds = generate_synthetic_dataset(
                movies, "C_moderate", seed, n_users=n_users, interactions_per_user=interactions_per_user,
                schema=schema, response_noise=sigma,
            )
            diag = trait_recovery_diagnostics(ds._latent_traits, ds.observed_profiles)
            rs.append(diag["mean_pearson_r"])
            maes.append(diag["overall_mae"])
            pers = RidgePersonalityModel(alpha=5.0).fit(ds.warm_train + ds.warm_val, ds.observed_profiles, ds.movie_features)
            rel_cold = _relevant_map(ds.cold_test_ratings)
            cold_pools, pop = _prepare_three_w3_pools(ds.warm_train + ds.warm_val, rel_cold, movies, vectors, settings)
            pu_c, _, cov_c, _ = _score_conditions(
                cold_pools, rel_cold,
                {"cold_correct_w15": (ds.observed_profiles, DIAGNOSTIC_ACTIVE_WEIGHT, True, False)},
                ds.movie_features, pers, movies, vectors, pop, key_prefix=f"s{seed}:",
            )
            pu_acc.update(pu_c["cold_correct_w15"])
            cov_acc.update(cov_c["cold_correct_w15"])
        ablations["noise"].append({
            "sigma": sigma,
            "mean_trait_r": sum(rs) / len(rs),
            "mean_trait_mae": sum(maes) / len(maes),
            **summarize_users(pu_acc, len(cov_acc), len(movies)),
        })

    for n_miss in (0, 1, 2, 3):
        pu_acc = {}
        cov_acc = set()
        rs, maes = [], []
        for seed in seeds:
            ds = generate_synthetic_dataset(
                movies, "C_moderate", seed, n_users=n_users, interactions_per_user=interactions_per_user, schema=schema,
            )
            masked = mask_dimensions(ds.observed_profiles, n_miss, seed=seed + 77)
            diag = trait_recovery_diagnostics(ds._latent_traits, masked)
            rs.append(diag["mean_pearson_r"])
            maes.append(diag["overall_mae"])
            pers = RidgePersonalityModel(alpha=5.0).fit(ds.warm_train + ds.warm_val, masked, ds.movie_features)
            rel_cold = _relevant_map(ds.cold_test_ratings)
            cold_pools, pop = _prepare_three_w3_pools(ds.warm_train + ds.warm_val, rel_cold, movies, vectors, settings)
            pu_c, _, cov_c, _ = _score_conditions(
                cold_pools, rel_cold,
                {"cold_masked_w15": (masked, DIAGNOSTIC_ACTIVE_WEIGHT, True, False)},
                ds.movie_features, pers, movies, vectors, pop, key_prefix=f"s{seed}:",
            )
            pu_acc.update(pu_c["cold_masked_w15"])
            cov_acc.update(cov_c["cold_masked_w15"])
        ablations["missing_dims"].append({
            "missing_dimensions": n_miss,
            "mean_trait_r": sum(rs) / len(rs),
            "mean_trait_mae": sum(maes) / len(maes),
            **summarize_users(pu_acc, len(cov_acc), len(movies)),
        })

    return {"scenarios": scenario_out, "ablations": ablations, "wall_seconds": time.perf_counter() - t0}


def run_track_b(outer_train, outer_test_rows, movies, vectors, settings, user_filter=None, seed=2026):
    """Track B: ML-1M architectural integration & null-control test with independent synthetic profiles."""
    t0 = time.perf_counter()
    schema = build_schema()
    movie_features = build_movie_attributes(movies, seed=seed)

    # Generate independent synthetic profiles for all ML-1M training users (null-control inputs only).
    all_uids = sorted({r[0] for r in outer_train})
    if user_filter:
        all_uids = [u for u in all_uids if u in user_filter]
    latent_indep, _, base_profiles = generate_population_profiles(all_uids, schema, seed=seed, response_noise=0.45)
    controls = build_control_profiles(all_uids, schema, seed=seed + 100, latent_override=latent_indep, base_profiles=base_profiles)

    # Inner validation weight selection (zero outer-test peeking).
    train_for_inner = [r for r in outer_train if r[0] in set(all_uids)] if user_filter else outer_train
    inner_train, val_rows = split_inner_chronological(train_for_inner)
    rel_val = _relevant_map(val_rows)
    inner_users = group_by_user(inner_train)
    rel_val = {u: rel for u, rel in rel_val.items() if inner_users.get(u)}
    val_pools, val_pop = _prepare_three_w3_pools(inner_train, rel_val, movies, vectors, settings)
    pers_inner = RidgePersonalityModel(alpha=5.0).fit(inner_train, controls["correct"], movie_features)
    selected_w, weight_sweep = _select_weight_on_inner(
        val_pools, rel_val, controls["correct"], movie_features, pers_inner, vectors, val_pop,
    )

    # Outer refit & single outer evaluation on fixed cf_popularity@200 pool.
    rel_outer = _relevant_map(outer_test_rows)
    if user_filter:
        rel_outer = {u: rel for u, rel in rel_outer.items() if u in user_filter}
    outer_pools, outer_pop = _prepare_three_w3_pools(train_for_inner, rel_outer, movies, vectors, settings)
    pers_outer = RidgePersonalityModel(alpha=5.0).fit(train_for_inner, controls["correct"], movie_features)

    active_w = selected_w if selected_w > 0.0 else DIAGNOSTIC_ACTIVE_WEIGHT
    conditions = {
        "three_w3_baseline": (None, 0.0, False, False),
        "personality_disabled": (controls["correct"], active_w, False, False),
        "personality_weight_zero": (controls["correct"], 0.0, True, False),
        "personality_selected": (controls["correct"], selected_w, True, False),
        "correct_w15": (controls["correct"], active_w, True, False),
        "noisy_w15": (controls["noisy"], active_w, True, False),
        "missing_w15": (controls["missing"], active_w, True, False),
        "shuffled_w15": (controls["shuffled"], active_w, True, False),
        "random_w15": (controls["random"], active_w, True, False),
    }
    per_user, summary, _, ranked_lists = _score_conditions(
        outer_pools, rel_outer, conditions, movie_features, pers_outer, movies, vectors, outer_pop,
    )

    # Bit-identical disabled/zero-weight verification & list stability diagnostics.
    base_lists = ranked_lists["three_w3_baseline"]
    stability = {
        cname: list_change_rate(base_lists, ranked_lists[cname])
        for cname in conditions if cname != "three_w3_baseline"
    }
    disabled_exact = (
        ranked_lists["three_w3_baseline"] == ranked_lists["personality_disabled"]
        and ranked_lists["three_w3_baseline"] == ranked_lists["personality_weight_zero"]
    )
    segments = segment_analysis(train_for_inner, per_user)
    pareto_inputs = {
        cname: {
            "ndcg": summary[cname]["ndcg"],
            "coverage": summary[cname]["coverage"],
            "ild": summary[cname]["ild"],
            "novelty": summary[cname]["novelty_p11"],
        }
        for cname in summary
    }
    front, dominated, pareto_report = pareto_front(pareto_inputs)
    return {
        "inner_selected_weight": selected_w,
        "diagnostic_active_weight": active_w,
        "weight_sweep": weight_sweep,
        "disabled_exact_match": disabled_exact,
        "summary": summary,
        "per_user": per_user,
        "stability": stability,
        "segments": segments,
        "pareto": {"front": front, "dominated": dominated, "report": pareto_report, "representatives": pick_representatives(pareto_inputs, front)},
        "wall_seconds": time.perf_counter() - t0,
        "users": len(outer_pools),
    }
