import os
import numpy as np
import torch
import matplotlib.pyplot as plt

from config.configurator import configs
from data_utils.build_data_handler import build_data_handler
from models.bulid_model import build_model


# ============================================================
# MAD 计算函数
# ============================================================

def compute_mad(emb: np.ndarray):
    """
    emb: [N, D] numpy array
    """
    # L2 normalize（论文惯例）
    emb = emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-8)

    # pairwise distance
    dist = np.linalg.norm(
        emb[:, None, :] - emb[None, :, :],
        axis=2
    )
    return dist.mean()


def mad_by_layer(layer_embeds, max_nodes=3000):
    """
    layer_embeds: list of torch.Tensor [N, D]
    """
    mad_list = []

    for l, emb in enumerate(layer_embeds):
        emb = emb.detach().cpu().numpy()

        # 随机采样，防止 OOM
        if emb.shape[0] > max_nodes:
            idx = np.random.choice(emb.shape[0], max_nodes, replace=False)
            emb = emb[idx]

        mad = compute_mad(emb)
        mad_list.append(mad)

    return mad_list


# ============================================================
# 加载模型 & 获取每一层 embedding
# ============================================================

def load_model_and_layer_embeds():
    """
    返回:
        user_layer_embeds: list of [n_users, dim]
        item_layer_embeds: list of [n_items, dim]
    """

    data_handler = build_data_handler()
    data_handler.load_data()

    model = build_model(data_handler).to(configs['device'])

    ckpt_path = f"./encoder/checkpoint/{configs['model']['name']}/" \
                f"{configs['model']['name']}-{configs['data']['name']}-2023.pth"

    print("[Load checkpoint]", ckpt_path)
    ckpt = torch.load(ckpt_path, map_location=configs['device'])
    model.load_state_dict(ckpt, strict=True)

    model.eval()

    n_users = model.user_num

    with torch.no_grad():

        # ===== LightGCN 风格 =====
        if configs['model']['name'] == 'lightgcn':
            # forward 需要支持 return_all_layers
            layer_embeds = model.forward(
                model.adj,
                keep_rate=1.0,
                return_all_layers=True
            )

        # ===== DCCF / GCLDM 风格 =====
        else:
            # 假设 forward 返回 gnn_embeds
            _, _, gnn_embeds, *_ = model.forward()
            layer_embeds = gnn_embeds  # list of [N, D]

    user_layer_embeds = [emb[:n_users] for emb in layer_embeds]
    item_layer_embeds = [emb[n_users:] for emb in layer_embeds]

    return user_layer_embeds, item_layer_embeds


# ============================================================
# 可视化
# ============================================================

def plot_mad(user_mads, item_mads, save_name):
    layers = list(range(len(user_mads)))

    plt.figure(figsize=(6, 4))
    plt.plot(layers, user_mads, marker='o', label='User')
    plt.plot(layers, item_mads, marker='s', label='Item')

    plt.xlabel("GNN Layer")
    plt.ylabel("MAD")
    plt.title(f"MAD vs Layer Depth ({configs['model']['name'].upper()})")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(save_name, dpi=300)
    plt.close()

    print("[Saved]", save_name)


# ============================================================
# 主流程
# ============================================================

def main():
    print("===================================")
    print("[Model]  ", configs['model']['name'])
    print("[Dataset]", configs['data']['name'])
    print("===================================")

    user_layer_embeds, item_layer_embeds = load_model_and_layer_embeds()

    print("Computing User MAD...")
    user_mads = mad_by_layer(user_layer_embeds)

    print("Computing Item MAD...")
    item_mads = mad_by_layer(item_layer_embeds)

    for l in range(len(user_mads)):
        print(f"Layer {l}: User MAD={user_mads[l]:.4f}, Item MAD={item_mads[l]:.4f}")

    save_path = f"mad_{configs['model']['name']}_{configs['data']['name']}.png"
    plot_mad(user_mads, item_mads, save_path)

    print("=== MAD Experiment Finished ===")


if __name__ == "__main__":
    main()
