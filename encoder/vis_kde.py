import numpy as np
import torch
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.manifold import TSNE

from config.configurator import configs
from data_utils.build_data_handler import build_data_handler
from models.bulid_model import build_model


# ============================================================
# 第二版向量可视化NCL风格，效果不太好 
# ============================================================


# ============================================================
# 1. 加载 Item Embedding
# ============================================================
def load_item_embedding():
    print(f"[Model]   {configs['model']['name']}")
    print(f"[Dataset] {configs['data']['name']}")

    data_handler = build_data_handler()
    data_handler.load_data()

    model = build_model(data_handler).to(configs["device"])

    ckpt_path = (
        f"./encoder/checkpoint/"
        f"{configs['model']['name']}/"
        f"{configs['model']['name']}-{configs['data']['name']}-2023.pth"
    )

    print("[Load checkpoint]", ckpt_path)
    ckpt = torch.load(ckpt_path, map_location=configs["device"])
    model.load_state_dict(ckpt, strict=True)

    model.eval()
    with torch.no_grad():
        if configs["model"]["name"] == "lightgcn":
            _, item_emb = model.forward(model.adj, keep_rate=1.0)
        else:
            _, item_emb, *_ = model.forward()

    return item_emb.cpu().numpy()


# ============================================================
# 2. 自动采样
# ============================================================
def auto_sample(emb, max_points=5000):
    n = emb.shape[0]
    if n > max_points:
        print(f"[Sampling] {n} → {max_points}")
        idx = np.random.choice(n, max_points, replace=False)
        return emb[idx]
    return emb


# ============================================================
# 3. t-SNE
# ============================================================
def run_tsne(emb):
    tsne = TSNE(
        n_components=2,
        perplexity=30,
        init="pca",
        learning_rate="auto",
        n_iter=1000,
        random_state=42,
        verbose=1
    )
    return tsne.fit_transform(emb)


# ============================================================
# 4. KDE 可视化（论文 / NCL 风格）
# ============================================================
def plot_kde_paper_style(emb_2d):
    x = emb_2d[:, 0]
    y = emb_2d[:, 1]

    plt.figure(figsize=(4, 4))

    sns.kdeplot(
        x=x,
        y=y,
        fill=True,
        cmap="Blues",
        levels=20,        # 等高层数，控制细腻程度
        thresh=0.05,      # 去掉低密度噪声（关键）
        bw_adjust=1.2,    # 带宽，调大容易形成“环”
        linewidths=0      # 不强调轮廓线
    )

    # 论文风格：无坐标轴
    plt.axis("off")
    plt.gca().set_aspect("equal")

    save_path = (
        f"kde_item_{configs['model']['name']}_"
        f"{configs['data']['name']}.png"
    )

    plt.tight_layout(pad=0)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()

    print("[Saved]", save_path)


# ============================================================
# 5. 主流程
# ============================================================
def main():
    item_emb = load_item_embedding()
    item_emb = auto_sample(item_emb, max_points=5000)

    print("[t-SNE] Running...")
    emb_2d = run_tsne(item_emb)

    plot_kde_paper_style(emb_2d)

    print("=== Visualization Done ===")


if __name__ == "__main__":
    main()
