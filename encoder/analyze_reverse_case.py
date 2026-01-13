import argparse
import pickle
import numpy as np
from collections import deque
from scipy.spatial.distance import cosine
import os

from config.configurator import configs
from data_utils.build_data_handler import build_data_handler


# =========================
# Utils
# =========================
def cosine_sim(a, b):
    return 1.0 - cosine(a, b)


def load_user_profiles(dataset):
    with open(f"./data/{dataset}/usr_prf.pkl", "rb") as f:
        return pickle.load(f)


# =========================
# User-user hop distance
# =========================
def build_user_graph_dist(data_handler, start_user):
    trn_mat = data_handler.trn_mat.tocsr()  # [U, I]
    num_users, num_items = trn_mat.shape

    item_to_users = [[] for _ in range(num_items)]
    coo = trn_mat.tocoo()
    for u, i in zip(coo.row, coo.col):
        item_to_users[i].append(u)

    visited_users = {start_user}
    visited_items = set()

    q = deque()
    q.append((start_user, 0))
    user_dist = {start_user: 0}

    while q:
        u, d = q.popleft()
        for it in trn_mat[u].indices:
            if it in visited_items:
                continue
            visited_items.add(it)
            for v in item_to_users[it]:
                if v not in visited_users:
                    visited_users.add(v)
                    user_dist[v] = d + 1
                    q.append((v, d + 1))

    return user_dist


# =========================
# Main
# =========================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--target_user", type=int, required=True)
    parser.add_argument(
        "--compare_model",
        required=True,
        choices=["dccflk_gene", "dccflk_plus"]
    )
    parser.add_argument("--topk", type=int, default=10)
    args = parser.parse_args()

    dataset = args.dataset
    u = args.target_user
    compare_model = args.compare_model
    topk = args.topk
    base_model = "dccf_lk"

    # --------------------------------------------------
    # 1. Load embeddings
    # --------------------------------------------------
    base_emb = np.load(f"./encoder/tmp/{base_model}_{dataset}_u{u}.npz")["user_emb"]
    comp_emb = np.load(f"./encoder/tmp/{compare_model}_{dataset}_u{u}.npz")["user_emb"]

    # --------------------------------------------------
    # 2. Load data
    # --------------------------------------------------
    configs["data"]["name"] = dataset
    data_handler = build_data_handler()
    data_handler.load_data()

    # --------------------------------------------------
    # 3. User-user distance
    # --------------------------------------------------
    user_dist = build_user_graph_dist(data_handler, u)
    long_users = [v for v, d in user_dist.items() if d > 3]

    # --------------------------------------------------
    # 4. Similarity & rank
    # --------------------------------------------------
    base_sims = {}
    comp_sims = {}

    for v in long_users:
        base_sims[v] = cosine_sim(base_emb[u], base_emb[v])
        comp_sims[v] = cosine_sim(comp_emb[u], comp_emb[v])

    base_rank = {
        v: r for r, (v, _) in enumerate(
            sorted(base_sims.items(), key=lambda x: x[1], reverse=True), start=1
        )
    }
    comp_rank = {
        v: r for r, (v, _) in enumerate(
            sorted(comp_sims.items(), key=lambda x: x[1], reverse=True), start=1
        )
    }

    # --------------------------------------------------
    # 5. Reverse mining
    # --------------------------------------------------
    cases = []
    for v in long_users:
        cases.append({
            "user": v,
            "delta_rank": base_rank[v] - comp_rank[v],
            "base_rank": base_rank[v],
            "comp_rank": comp_rank[v],
            "base_sim": base_sims[v],
            "comp_sim": comp_sims[v],
        })

    cases.sort(key=lambda x: x["delta_rank"], reverse=True)
    top_cases = cases[:topk]

    # --------------------------------------------------
    # 6. Load profiles
    # --------------------------------------------------
    profiles = load_user_profiles(dataset)

    # --------------------------------------------------
    # 7. Save to TXT
    # --------------------------------------------------
    save_dir = "./encoder/tmp/caseStudy_result"
    os.makedirs(save_dir, exist_ok=True)

    save_path = os.path.join(
        save_dir,
        f"{base_model}_vs_{compare_model}_{dataset}_u{u}.txt"
    )

    with open(save_path, "w", encoding="utf-8") as f:
        f.write(f"# Case Study Result\n")
        f.write(f"# Dataset: {dataset}\n")
        f.write(f"# Target user: u{u}\n")
        f.write(f"# Base model: {base_model}\n")
        f.write(f"# Compare model: {compare_model}\n")
        f.write(f"# Long-distance users (>3 hops): {len(long_users)}\n\n")

        for i, c in enumerate(top_cases, 1):
            v = c["user"]
            f.write(f"--- Case {i}: u{u} vs u{v} ---\n")
            f.write(
                f"Base rank ({base_model}): {c['base_rank']}\n"
                f"Rank ({compare_model}): {c['comp_rank']}\n"
                f"Delta rank: +{c['delta_rank']}\n"
            )
            f.write(
                f"Base sim: {c['base_sim']:.4f}\n"
                f"{compare_model} sim: {c['comp_sim']:.4f}\n"
            )

            if v in profiles:
                f.write("\n[User Profile]\n")
                f.write(profiles[v]["profile"].strip() + "\n")
                f.write("\n[Reasoning]\n")
                f.write(profiles[v]["reasoning"].strip() + "\n")

            f.write("\n" + "=" * 60 + "\n\n")

    print(f"[Saved] Case study results written to:\n{save_path}")


if __name__ == "__main__":
    main()
