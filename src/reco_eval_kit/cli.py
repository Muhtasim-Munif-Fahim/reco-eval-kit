"""End-to-end CLI: generate, split, recommend, evaluate, and report."""

from __future__ import annotations

import argparse

import numpy as np

from .baselines import (
    BPRRecommender,
    ItemItemCooccurrenceRecommender,
    ItemKNNRecommender,
    PopularityRecommender,
    RandomRecommender,
    UserKNNRecommender,
    PureSVDRecommender,
    WRMFRecommender,
    EASERecommender,
    SlimRecommender,
    NMFRecommender,
    Item2VecRecommender,
    FactorizationMachineRecommender,
    SVDPlusPlusRecommender,
    RP3betaRecommender,
)
from .beyond_accuracy import (
    catalog_coverage,
    intra_list_diversity,
    novelty,
    profile_unexpectedness,
    serendipity,
    unexpectedness,
)
from .metrics import (
    average_precision_at_k,
    err_at_k,
    rbp_at_k,
    hit_rate_at_k,
    mean_inverse_popularity_at_k,
    mean_metric,
    mrr,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
)
from .diversify import dpp_rerank, mmr_rerank, rank_scores
from .report import render_report, save_report
from .significance import paired_comparison, per_user_scores, render_significance_section
from .splitting import leave_one_out
from .sampled import corrected_sampled_metric, full_rank
from .unbiased import mean_unbiased_metric_at_k, power_law_propensities
from .synthetic import generate_interactions, generate_item_features

CUTOFF_METRICS = (
    ("Precision@K", precision_at_k),
    ("Recall@K", recall_at_k),
    ("MAP@K", average_precision_at_k),
    ("NDCG@K", ndcg_at_k),
    ("ERR@K", err_at_k),
    ("RBP@K", rbp_at_k),
    ("HitRate@K", hit_rate_at_k),
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="reco-eval-kit",
        description="Evaluate baseline recommenders on synthetic implicit feedback.",
    )
    parser.add_argument("--n-users", type=int, default=100)
    parser.add_argument("--n-items", type=int, default=300)
    parser.add_argument("--n-interactions", type=int, default=6000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--output", default="evaluation_report.md")
    parser.add_argument(
        "--significance-baseline",
        default=None,
        metavar="MODEL",
        help="add paired randomization tests of NDCG@K for every model against this "
        "model name (for example 'popularity'), Holm-adjusted",
    )
    parser.add_argument(
        "--significance-resamples",
        type=int,
        default=2000,
        help="sign-flip / bootstrap resamples for --significance-baseline (default: 2000)",
    )
    parser.add_argument(
        "--propensity-gamma",
        type=float,
        default=None,
        metavar="GAMMA",
        help="add SNIPS-Recall@K / SNIPS-NDCG@K with power-law popularity "
        "propensities p ~ n**((gamma+1)/2) (Yang et al. 2018); for example 0.5",
    )
    parser.add_argument(
        "--diversify",
        choices=("mmr", "dpp"),
        default=None,
        help="also evaluate every model re-ranked for diversity: MMR (Carbonell & "
        "Goldstein 1998) or greedy DPP MAP (Chen et al. 2018) over its top "
        "--diversify-pool * k candidates, using item-feature cosine similarity",
    )
    parser.add_argument(
        "--diversify-lambda",
        type=float,
        default=0.7,
        help="relevance weight: MMR lambda in [0, 1] or DPP theta in [0, 1) (default: 0.7)",
    )
    parser.add_argument(
        "--diversify-pool",
        type=int,
        default=3,
        help="candidate pool size as a multiple of k for --diversify (default: 3)",
    )
    parser.add_argument(
        "--sampled-negatives",
        type=int,
        default=None,
        metavar="M",
        help="also report the common sampled protocol (rank the held-out item among M "
        "random unseen items) and its Krichene & Rendle (2020) bias-corrected "
        "estimates of the full-catalog HitRate@K / NDCG@K; for example 100",
    )
    parser.add_argument(
        "--sampled-gamma",
        type=float,
        default=0.01,
        help="variance penalty of the bias-variance correction (default: 0.01)",
    )
    return parser


def _sampled_columns(model, test_users, history_lists, relevant_lists, n_items, m, k, gamma, rng):
    """Naive and corrected sampled HitRate@K / NDCG@K for one fitted model."""
    ranks, n_neg = [], []
    catalog = range(n_items)
    for user, history, relevant in zip(test_users, history_lists, relevant_lists):
        positive = next(iter(relevant))
        pool = [i for i in catalog if i not in history and i != positive]
        if len(pool) < m:
            raise SystemExit(f"--sampled-negatives {m} exceeds the {len(pool)} unseen items of a user")
        sampled = rng.choice(pool, size=m, replace=False).tolist()
        ranking = model.recommend(user, k=n_items, exclude=history)
        ranks.append(full_rank(ranking, positive, sampled + [positive]))
        n_neg.append(len(pool))
    out = {}
    for label, metric in (("HR", "hit"), ("NDCG", "ndcg")):
        est = corrected_sampled_metric(ranks, n_neg, m, metric, k, method="bv", gamma=gamma)
        out[f"Sampled-{label}@K"] = est.naive
        out[f"Corrected-{label}@K"] = est.corrected
    return out


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.diversify is not None:
        upper = 1.0 if args.diversify == "mmr" else 1.0 - 1e-9
        if not 0.0 <= args.diversify_lambda <= upper:
            parser.error("--diversify-lambda must be in [0, 1] for mmr and [0, 1) for dpp")
        if args.diversify_pool < 1:
            parser.error("--diversify-pool must be >= 1")
    if args.sampled_negatives is not None and args.sampled_negatives < 1:
        parser.error("--sampled-negatives must be >= 1")
    if args.sampled_gamma < 0:
        parser.error("--sampled-gamma must be >= 0")

    interactions = generate_interactions(
        n_users=args.n_users,
        n_items=args.n_items,
        n_interactions=args.n_interactions,
        seed=args.seed,
    )
    train, test = leave_one_out(interactions)
    test_users = sorted(test["user_id"].unique())
    seen_by_user = train.groupby("user_id")["item_id"].agg(set).to_dict()
    relevant_by_user = test.groupby("user_id")["item_id"].agg(set).to_dict()
    popularity_counts = train["item_id"].value_counts().to_dict()
    item_features = generate_item_features(n_items=args.n_items, seed=args.seed + 1)
    history_lists = [seen_by_user.get(user, set()) for user in test_users]
    relevant_lists = [relevant_by_user[user] for user in test_users]

    models = {
        "popularity": PopularityRecommender(),
        f"random(seed={args.seed})": RandomRecommender(seed=args.seed),
        "item-item-cooccurrence": ItemItemCooccurrenceRecommender(),
        "item-knn(cosine)": ItemKNNRecommender(similarity="cosine"),
        "user-knn(cosine)": UserKNNRecommender(similarity="cosine"),
        f"bpr(seed={args.seed})": BPRRecommender(seed=args.seed),
        "puresvd": PureSVDRecommender(n_factors=16),
        f"wrmf(seed={args.seed})": WRMFRecommender(n_factors=16, n_epochs=8, seed=args.seed),
        "ease": EASERecommender(l2=200.0),
        "rp3beta": RP3betaRecommender(alpha=1.0, beta=0.5),
        "slim": SlimRecommender(l1_reg=0.05, l2_reg=0.1, n_iter=15),
        f"nmf(seed={args.seed})": NMFRecommender(n_factors=16, n_epochs=30, seed=args.seed),
        f"item2vec(seed={args.seed})": Item2VecRecommender(embedding_dim=16, n_epochs=3, seed=args.seed),
        f"fm(seed={args.seed})": FactorizationMachineRecommender(n_factors=16, n_epochs=20, seed=args.seed),
        f"svdpp(seed={args.seed})": SVDPlusPlusRecommender(n_factors=16, n_epochs=20, seed=args.seed),
    }

    if args.significance_baseline is not None and args.significance_baseline not in models:
        raise SystemExit(
            f"unknown --significance-baseline {args.significance_baseline!r}; "
            f"choose one of: {', '.join(models)}"
        )

    propensities = None
    if args.propensity_gamma is not None:
        propensities = power_law_propensities(popularity_counts, gamma=args.propensity_gamma)
        floor = min(propensities.values()) if propensities else 1e-3

    def score_lists(rec_lists):
        """Every report metric for one set of per-user recommendation lists."""
        scores = {
            label: mean_metric(metric, rec_lists, relevant_lists, k=args.k)
            for label, metric in CUTOFF_METRICS
        }
        scores["MRR"] = mean_metric(mrr, rec_lists, relevant_lists)
        scores["InversePopularity@K"] = mean_inverse_popularity_at_k(
            rec_lists, popularity_counts, args.k
        )
        scores["CatalogCoverage"] = catalog_coverage(rec_lists, args.n_items)
        scores["NoveltyBits"] = novelty(rec_lists, popularity_counts)
        scores["IntraListDiversity"] = intra_list_diversity(rec_lists, item_features)
        scores["Unexpectedness"] = unexpectedness(rec_lists, popularity_counts)
        scores["ProfileUnexpectedness"] = profile_unexpectedness(
            rec_lists, history_lists, item_features
        )
        scores["Serendipity"] = serendipity(
            rec_lists, relevant_lists, item_popularity=popularity_counts, k=args.k
        )
        scores["ProfileSerendipity"] = serendipity(
            rec_lists,
            relevant_lists,
            user_histories=history_lists,
            item_features=item_features,
            k=args.k,
        )
        if propensities is not None:
            for label, gain in (("SNIPS-Recall@K", "hit"), ("SNIPS-NDCG@K", "dcg")):
                scores[label] = mean_unbiased_metric_at_k(
                    rec_lists,
                    relevant_lists,
                    propensities,
                    args.k,
                    gain=gain,
                    default_propensity=floor,
                ).snips
        return scores

    results = {}
    ndcg_by_model = {}
    for name, model in models.items():
        model.fit(train)
        rec_lists = [
            model.recommend(user, k=args.k, exclude=history)
            for user, history in zip(test_users, history_lists)
        ]
        results[name] = score_lists(rec_lists)
        if args.sampled_negatives is not None:
            results[name].update(
                _sampled_columns(
                    model, test_users, history_lists, relevant_lists, args.n_items,
                    args.sampled_negatives, args.k, args.sampled_gamma,
                    np.random.default_rng(args.seed + 7),
                )
            )
        ndcg_by_model[name] = per_user_scores(ndcg_at_k, rec_lists, relevant_lists, k=args.k)
        if args.diversify is not None:
            pool = args.k * args.diversify_pool
            rerank = mmr_rerank if args.diversify == "mmr" else dpp_rerank
            reranked = []
            for user, history in zip(test_users, history_lists):
                cands = model.recommend(user, k=pool, exclude=history)
                reranked.append(
                    rerank(
                        cands,
                        rank_scores(len(cands)),
                        args.k,
                        args.diversify_lambda,
                        item_features=item_features,
                    )
                )
            div_name = f"{name}+{args.diversify}"
            results[div_name] = score_lists(reranked)
            ndcg_by_model[div_name] = per_user_scores(ndcg_at_k, reranked, relevant_lists, k=args.k)

    summary = {
        "Protocol": "leave-one-out",
        "Users evaluated": len(test_users),
        "Items in catalog": args.n_items,
        "Train interactions": len(train),
        "Test interactions": len(test),
        "Cutoff k": args.k,
        "Seed": args.seed,
    }
    if args.propensity_gamma is not None:
        summary["Propensity model"] = f"power law, gamma={args.propensity_gamma:g}"
    if args.sampled_negatives is not None:
        summary["Sampled protocol"] = (
            f"{args.sampled_negatives} negatives, bias-variance correction gamma={args.sampled_gamma:g}"
        )
    if args.diversify is not None:
        summary["Diversification"] = (
            f"{args.diversify}, lambda={args.diversify_lambda:g}, "
            f"pool={args.diversify_pool}xk"
        )
    markdown = render_report(results, dataset_summary=summary)
    if args.significance_baseline is not None:
        base = args.significance_baseline
        comparisons = {
            name: paired_comparison(
                ndcg_by_model[base],
                scores,
                n_resamples=args.significance_resamples,
                seed=args.seed,
            )
            for name, scores in ndcg_by_model.items()
            if name != base
        }
        markdown += "\n" + render_significance_section(comparisons, base, f"NDCG@{args.k}")
    output_path = save_report(markdown, args.output)

    print(f"evaluated {len(models)} recommenders on {len(test_users)} test users")
    for name, scores in results.items():
        print(f"{name}: NDCG@{args.k}={scores['NDCG@K']:.4f} MRR={scores['MRR']:.4f}")
    print(f"report written to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())