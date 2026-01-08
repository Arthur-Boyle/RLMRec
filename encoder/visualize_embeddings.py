# ============================================================
# 强制限制 BLAS 多线程，避免 t-SNE segmentation fault
# ============================================================
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

from config.configurator import configs
from data_utils.build_data_handler import build_data_handler
from models.bulid_model import build_model


# ============================================================
# 加载模型与 embeddings
# ============================================================
def load_trained_model():
    """加载训练好的 LightGCN 模型并返回用户和物品 Embedding"""

    data_handler = build_data_handler()
    data_handler.load_data()

    model = build_model(data_handler).to(configs['device'])

    ckpt_path = "/root/ryw/Rec/RLMRec/encoder/checkpoint/dccf_lk/dccf_lk-yelp-2023.pth"
    print("Loading checkpoint:", ckpt_path)
    ckpt = torch.load(ckpt_path, map_location=configs['device'])
    model.load_state_dict(ckpt, strict=True)

    model.eval()
    print(configs['model']['name'], configs['data']['name'])

    with torch.no_grad():
        #lightgcn的版本
        #user_emb, item_emb = model.forward(model.adj, keep_rate=1.0)

        #dccf_lk的版本
        user_emb, item_emb, gnn_embeds, int_embeds, gaa_embeds, iaa_embeds = model.forward()

    return user_emb.cpu().numpy(), item_emb.cpu().numpy()


# ============================================================
# 采样函数
# ============================================================
def auto_sample(emb, max_points=8000):
    """t-SNE 非常耗资源，自动采样避免崩溃"""
    n = emb.shape[0]
    if n > max_points:
        print(f"[Sampling] embedding 过大 ({n})，随机采样 {max_points} 个点用于 t-SNE。")
        idx = np.random.choice(n, max_points, replace=False)
        return emb[idx], idx
    return emb, None


# ============================================================
# 通用可视化函数
# ============================================================
def visualize_2d(emb2d, labels, title, save_path):
    plt.figure(figsize=(6, 6))
    plt.scatter(emb2d[:, 0], emb2d[:, 1], s=2, c=labels, cmap='Spectral')
    plt.title(title)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print("Saved:", save_path)


# ============================================================
# 降维函数
# ============================================================
def reduce_pca(emb):
    pca = PCA(n_components=2)
    return pca.fit_transform(emb)


def reduce_tsne(emb):
    tsne = TSNE(
        n_components=2,
        learning_rate='auto',
        init='pca',
        perplexity=30,
        n_iter=1000,
        verbose=1
    )
    return tsne.fit_transform(emb)


# ============================================================
# 主流程
# ============================================================
def main():
    user_emb, item_emb = load_trained_model()

    # 1) 混合 Embedding 用于整体可视化
    all_emb = np.concatenate([user_emb, item_emb], axis=0)
    all_labels = np.array([0] * user_emb.shape[0] + [1] * item_emb.shape[0])

    # ============================================================
    # PCA 可视化（稳定且快速）
    # ============================================================
    print("Running PCA...")
    all_pca = reduce_pca(all_emb)
    visualize_2d(all_pca, all_labels, "PCA User+Item Embeddings", "pca_all.png")

    # ============================================================
    # t-SNE 可视化
    # ============================================================
    print("Running t-SNE (可能很慢)...")

    # 自动采样 - 避免 t-SNE 直接 OOM 或 segmentation fault
    all_emb_sampled, sampled_idx = auto_sample(all_emb, max_points=8000)
    if sampled_idx is not None:
        tsne_labels = all_labels[sampled_idx]
    else:
        tsne_labels = all_labels

    all_tsne = reduce_tsne(all_emb_sampled)
    visualize_2d(all_tsne, tsne_labels, "t-SNE User+Item Embeddings", "tsne_all.png")

    # ============================================================
    # 分别绘制 User / Item Embedding 的 PCA
    # ============================================================
    print("Drawing User and Item PCA...")

    user_pca = reduce_pca(user_emb)
    visualize_2d(user_pca, np.zeros(user_pca.shape[0]), "PCA Users", "pca_user.png")

    item_pca = reduce_pca(item_emb)
    visualize_2d(item_pca, np.ones(item_pca.shape[0]), "PCA Items", "pca_item.png")

    print("=== Visualization Completed ===")


if __name__ == "__main__":
    main()
