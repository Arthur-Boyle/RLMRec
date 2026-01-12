import argparse
import pickle
import numpy as np
from scipy.spatial.distance import cosine


def cosine_sim(a, b):
    return 1.0 - cosine(a, b)


def load_profiles(dataset):
    with open(f"./data/{dataset}/usr_prf.pkl", "rb") as f:
        return pickle.load(f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--target_user", type=int, required=True)
    parser.add_argument("--topk", type=int, default=10)
    args = parser.parse_args()

    u = args.target_user
    dataset = args.dataset

    base = np.load(f"./encoder/tmp/dccf_lk_{dataset}_u{u}.npz")
    gene = np.load(f"./encoder/tmp/dccflk_gene_{dataset}_u{u}.npz")

    user_emb_base = base["user_emb"]
    user_emb_gene = gene["user_emb"]
    long_users = base["long_users"]

    base_sims = {}
    gene_sims = {}

    for v in long_users:
        base_sims[v] = cosine_sim(user_emb_base[u], user_emb_base[v])
        gene_sims[v] = cosine_sim(user_emb_gene[u], user_emb_gene[v])

    base_rank = {
        v: r for r, (v, _) in enumerate(
            sorted(base_sims.items(), key=lambda x: x[1], reverse=True), 1
        )
    }
    gene_rank = {
        v: r for r, (v, _) in enumerate(
            sorted(gene_sims.items(), key=lambda x: x[1], reverse=True), 1
        )
    }

    cases = []
    for v in long_users:
        cases.append({
            "user": v,
            "delta_rank": base_rank[v] - gene_rank[v],
            "base_rank": base_rank[v],
            "gene_rank": gene_rank[v],
            "base_sim": base_sims[v],
            "gene_sim": gene_sims[v],
        })

    cases.sort(key=lambda x: x["delta_rank"], reverse=True)
    top_cases = cases[:args.topk]

    profiles = load_profiles(dataset)

    print("\n===== Reverse Case Study Results =====")
    print(f"Target user: u{u}\n")

    for i, c in enumerate(top_cases, 1):
        v = c["user"]
        print(f"--- Case {i}: u{u} vs u{v} ---")
        print(
            f"Base rank: {c['base_rank']}, "
            f"Gene rank: {c['gene_rank']}, "
            f"Δrank: +{c['delta_rank']}"
        )
        print(
            f"Base sim: {c['base_sim']:.4f}, "
            f"Gene sim: {c['gene_sim']:.4f}"
        )

        if v in profiles:
            print("\n[User Profile]")
            print(profiles[v]["profile"][:300], "...")
            print("\n[Reasoning]")
            print(profiles[v]["reasoning"][:300], "...")
        print()


if __name__ == "__main__":
    main()
