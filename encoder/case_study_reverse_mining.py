import argparse
import pickle
import torch
import numpy as np
from tqdm import tqdm
from scipy.spatial.distance import cosine

from config.configurator import configs
from data_utils.build_data_handler import build_data_handler
from models.bulid_model import build_model


# ===============================
# Utils
# ===============================
def cosine_sim(a, b):
    return 1.0 - cosine(a, b)


def load_user_profiles(dataset):
    path = f"./data/{dataset}/usr_prf.pkl"
    with open(path, "rb") as f:
        return pickle.load(f)


# ===============================
# Load model + layer-0 user embeds
# ===============================
def load_user_embeds_layer0(model_name, dataset, device):
    """
    Use configs + yml to correctly initialize model hyper-parameters
    """

    # -------- set configs (trigger yml loading) --------
    configs["model"]["name"] = model_name
    configs["data"]["name"] = dataset
    configs["device"] = device

    # ⚠️ 关键：让 configs 完成内部初始化（你工程里一定有）
    configs._init_config()

    # -------- data --------
    data_handler = build_data_handler()
    data_handler.load_data()

    # -------- model --------
    model = build_model(data_handler).to(device)

    ckpt_path = (
        f"./encoder/checkpoint/{model_name}/"
        f"{model_name}-{dataset}-2023.pth"
    )
    print("[Load checkpoint]", ckpt_path)

    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt, strict=True)
    model.eval()

    n_users = model.user_num

    with torch.no_grad():
        _, _, gnn_embeds, *_ = model.forward()

    # Layer-0 (semantic / ID initialization)
    emb = gnn_embeds[0].detach().cpu().numpy()

    return emb[:n_users], data_handler


# ===============================
# Main: Reverse Case Study Mining
# ===============================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--target_user", type=int, required=True)
    parser.add_argument("--topk", type=int, default=10)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    dataset = args.dataset
    u = args.target_user
    device = args.device

    # -------- load baseline --------
    user_emb_base, data_handler = load_user_embeds_layer0(
        model_name="dccf_lk",
        dataset=dataset,
        device=device
    )

    # -------- load enhanced --------
    user_emb_gene, _ = load_user_embeds_layer0(
        model_name="dccflk_gene",
        dataset=dataset,
        device=device
    )

    # -------- graph distance --------
    user_dist = data_handler.user_graph_dist[u]
    long_users = [v for v, d in user_dist.items() if d > 3]

    print(f"# Long-distance users (>3 hops): {len(long_users)}")

    # -------- similarity --------
    base_sims = {}
    gene_sims = {}

    for v in tqdm(long_users):
        base_sims[v] = cosine_sim(user_emb_base[u], user_emb_base[v])
        gene_sims[v] = cosine_sim(user_emb_gene[u], user_emb_gene[v])

    # -------- ranking --------
    base_rank = {
        v: r for r, (v, _) in enumerate(
            sorted(base_sims.items(), key=lambda x: x[1], reverse=True), start=1
        )
    }
    gene_rank = {
        v: r for r, (v, _) in enumerate(
            sorted(gene_sims.items(), key=lambda x: x[1], reverse=True), start=1
        )
    }

    # -------- improvement --------
    improvements = []
    for v in long_users:
        improvements.append({
            "user": v,
            "delta_rank": base_rank[v] - gene_rank[v],
            "delta_sim": gene_sims[v] - base_sims[v],
            "base_rank": base_rank[v],
            "gene_rank": gene_rank[v],
            "base_sim": base_sims[v],
            "gene_sim": gene_sims[v],
        })

    improvements.sort(key=lambda x: x["delta_rank"], reverse=True)
    top_cases = improvements[:args.topk]

    # -------- profiles --------
    profiles = load_user_profiles(dataset)

    print("\n===== Reverse Case Study Results =====")
    print(f"Target user: u{u}\n")

    for i, case in enumerate(top_cases, 1):
        v = case["user"]
        print(f"--- Case {i}: u{u} vs u{v} ---")
        print(
            f"Base rank: {case['base_rank']}, "
            f"Gene rank: {case['gene_rank']}, "
            f"Δrank: +{case['delta_rank']}"
        )
        print(
            f"Base sim: {case['base_sim']:.4f}, "
            f"Gene sim: {case['gene_sim']:.4f}, "
            f"Δsim: {case['delta_sim']:.4f}"
        )

        if v in profiles:
            print("\n[User Profile]")
            print(profiles[v]["profile"][:300], "...")
            print("\n[Reasoning]")
            print(profiles[v]["reasoning"][:300], "...")
        print("\n")


if __name__ == "__main__":
    main()
