import os
import argparse
import torch
import numpy as np
import scipy.sparse as sp
from collections import deque

from config.configurator import configs
from data_utils.build_data_handler import build_data_handler
from models.bulid_model import build_model


# =====================================================
# Utils
# =====================================================
def cosine_sim(a, b):
    return float(
        np.dot(a, b) /
        (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12)
    )


def build_bipartite_adj(trn_mat):
    """
    Build adjacency list for user-item bipartite graph
    """
    n_users, n_items = trn_mat.shape
    total_nodes = n_users + n_items
    adj = [[] for _ in range(total_nodes)]

    coo = trn_mat.tocoo()
    for u, i in zip(coo.row, coo.col):
        v = n_users + i
        adj[u].append(v)
        adj[v].append(u)

    return adj, n_users


def bfs_user_distances(adj, n_users, start_u):
    """
    BFS from start user, return distance to all users
    """
    dist = np.full(len(adj), np.inf)
    q = deque([start_u])
    dist[start_u] = 0

    while q:
        x = q.popleft()
        for y in adj[x]:
            if not np.isfinite(dist[y]):
                dist[y] = dist[x] + 1
                q.append(y)

    return dist[:n_users]


def load_user_layer_embeds():
    """
    Return:
        user_layer_embeds: list of [n_users, dim] numpy arrays
    """
    data_handler = build_data_handler()
    data_handler.load_data()

    model = build_model(data_handler).to(configs['device'])

    ckpt_path = (
        f"./encoder/checkpoint/{configs['model']['name']}/"
        f"{configs['model']['name']}-{configs['data']['name']}-2023.pth"
    )

    print("[Load checkpoint]", ckpt_path)
    ckpt = torch.load(ckpt_path, map_location=configs['device'])
    model.load_state_dict(ckpt, strict=True)
    model.eval()

    n_users = model.user_num

    with torch.no_grad():
        _, _, gnn_embeds, *_ = model.forward()

    user_layer_embeds = []
    for l, emb in enumerate(gnn_embeds):
        emb = emb.detach().cpu().numpy()
        user_layer_embeds.append(emb[:n_users])
        print(f"[Layer {l}] user_emb shape = {user_layer_embeds[-1].shape}")

    return user_layer_embeds, data_handler


# =====================================================
# Main
# =====================================================
def main():
    parser = argparse.ArgumentParser("Case Study: Long-distance User Similarity")
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--model", type=str, required=True,
                        help="dccf_lk or dccflk_gene")
    parser.add_argument("--target_user", type=int, required=True)
    parser.add_argument("--other_user", type=int, required=True)
    parser.add_argument("--hop", type=int, default=3,
                        help="long-distance threshold")
    args = parser.parse_args()

    # -------------------------------------------------
    # Set configs
    # -------------------------------------------------
    configs['data']['name'] = args.dataset
    configs['model']['name'] = args.model

    # -------------------------------------------------
    # Load embeddings
    # -------------------------------------------------
    user_layer_embeds, data_handler = load_user_layer_embeds()

    # -------------------------------------------------
    # Build graph & distances
    # -------------------------------------------------
    trn_mat = data_handler.trn_mat.tocsr()
    adj, n_users = build_bipartite_adj(trn_mat)
    user_dist = bfs_user_distances(adj, n_users, args.target_user)

    long_users = np.where(
        (user_dist > args.hop) & np.isfinite(user_dist)
    )[0]

    print(f"# Long-distance users (> {args.hop} hops): {len(long_users)}")

    assert args.other_user in long_users, \
        f"user {args.other_user} is not >{args.hop} hops away"

    # -------------------------------------------------
    # Case Study per layer
    # -------------------------------------------------
    print("\n===== Case Study Result =====")
    print(f"Target user: u{args.target_user}")
    print(f"Other user : u{args.other_user}\n")

    for l, emb in enumerate(user_layer_embeds):
        u_vec = emb[args.target_user]
        sims = np.array([cosine_sim(u_vec, emb[v]) for v in long_users])

        rank = int(np.sum(sims > sims[long_users.tolist().index(args.other_user)]) + 1)
        sim_uv = cosine_sim(u_vec, emb[args.other_user])

        print(
            f"[Layer {l}] "
            f"sim(u{args.target_user}, u{args.other_user}) = {sim_uv:.4f}, "
            f"rank = {rank}/{len(long_users)}"
        )


if __name__ == "__main__":
    main()
