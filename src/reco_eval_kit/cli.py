"""End-to-end CLI: generate, split, recommend, evaluate, and report."""

from __future__ import annotations

import argparse

from .baselines import (
    BPRRecommender,
    ItemItemCooccurrenceRecommender,
    PopularityRecommender,
    RandomRecommender,
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
    hit_rate_at_k,
    mean_metric,
    mrr,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
)
from .report import render_report, save_report
from .splitting import leave_one_out
from .synthetic import generate_interactions, generate_item_features

CUTOFF_METRICS = (
    ("Precision@K", precision_at_k),
    ("Recall@K", recall_at_k),
    ("MAP@K", average_precision_at_k),
    ("NDCG@K", ndcg_at_k),
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
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)

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
        f"bpr(seed={args.seed})": BPRRecommender(seed=args.seed),
    }

    results = {}
    for name, model in models.items():
        model.fit(train)
        rec_lists = [
            model.recommend(user, k=args.k, exclude=history)
            for user, history in zip(test_users, history_lists)
        ]

        scores = {
            label: mean_metric(metric, rec_lists, relevant_lists, k=args.k)
            for label, metric in CUTOFF_METRICS
        }
        scores["MRR"] = mean_metric(mrr, rec_lists, relevant_lists)
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
        results[name] = scores

    summary = {
        "Protocol": "leave-one-out",
        "Users evaluated": len(test_users),
        "Items in catalog": args.n_items,
        "Train interactions": len(train),
        "Test interactions": len(test),
        "Cutoff k": args.k,
        "Seed": args.seed,
    }
    markdown = render_report(results, dataset_summary=summary)
    output_path = save_report(markdown, args.output)

    print(f"evaluated {len(models)} recommenders on {len(test_users)} test users")
    for name, scores in results.items():
        print(f"{name}: NDCG@{args.k}={scores['NDCG@K']:.4f} MRR={scores['MRR']:.4f}")
    print(f"report written to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())