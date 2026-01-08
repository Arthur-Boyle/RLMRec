import torch
import numpy as np
import pandas as pd
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
import random

from config.configurator import configs
from data_utils.build_data_handler import build_data_handler
from models.bulid_model import build_model



# ============================================================
# 加载模型与 embeddings
# ============================================================
def load_trained_model_lightgcn():
    """加载训练好的 LightGCN 模型并返回用户和物品 Embedding"""

    data_handler = build_data_handler()
    data_handler.load_data()

    model = build_model(data_handler).to(configs['device'])

    ckpt_path = "/root/ryw/Rec/RLMRec/encoder/checkpoint/lightgcn/lightgcn-amazon-2023.pth"
    print("Loading checkpoint:", ckpt_path)
    ckpt = torch.load(ckpt_path, map_location=configs['device'])
    model.load_state_dict(ckpt, strict=True)

    model.eval()
    with torch.no_grad():
        user_emb, item_emb = model.forward(model.adj, keep_rate=1.0)

    return user_emb.cpu().numpy(), item_emb.cpu().numpy()

def load_trained_model_gcldm():
    """加载训练好的 GCLDM 模型并返回用户和物品 Embedding"""

    data_handler = build_data_handler()
    data_handler.load_data()

    model = build_model(data_handler).to(configs['device'])

    ckpt_path = "/root/ryw/Rec/RLMRec/encoder/checkpoint/dccf_lk/dccf_lk-amazon-2023.pth"
    print("Loading checkpoint:", ckpt_path)
    ckpt = torch.load(ckpt_path, map_location=configs['device'])
    model.load_state_dict(ckpt, strict=True)

    model.eval()
    with torch.no_grad():
        gcldm_user_emb, gcldm_item_emb, gnn_embeds, int_embeds, gaa_embeds, iaa_embeds = model.forward()

    return gcldm_user_emb.cpu().numpy(), gcldm_item_emb.cpu().numpy()

# ==============================================================================
# 2. 核心可视化逻辑 (Visualization Logic)
# ==============================================================================

def tsne_reduction(embeddings, num_samples=5000):
    """
    对嵌入向量进行采样和 t-SNE 降维。
    
    参数:
    embeddings (np.array): 模型的嵌入向量 (N, D)
    num_samples (int): 采样的节点数量
    
    返回:
    pd.DataFrame: 包含 2D 坐标的 DataFrame
    """
    
    # 1. 采样
    N, D = embeddings.shape
    if N > num_samples:
        indices = np.random.choice(N, num_samples, replace=False)
        sample_embeddings = embeddings[indices]
    else:
        sample_embeddings = embeddings
        
    print(f"Running t-SNE on {sample_embeddings.shape} samples (Dim: {D}). This may take a moment.")

    # 2. 降维 (使用 t-SNE)
    # Perplexity 和 learning_rate 等参数可以根据数据集大小进行微调
    tsne = TSNE(n_components=2, 
                random_state=42, 
                perplexity=30, 
                learning_rate='auto', 
                init='pca', 
                n_jobs=-1)
    
    embeddings_2d = tsne.fit_transform(sample_embeddings)
    
    df = pd.DataFrame(embeddings_2d, columns=['Dim 1', 'Dim 2'])
    return df

def plot_kde_distribution(df_gcl, df_base, item_or_user):
    """
    使用核密度估计 (KDE) 绘制两个模型的嵌入分布对比图 [2]。
    """
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.figure(figsize=(12, 6))

    # --- Plot GCLDM ---
    plt.subplot(1, 2, 1)
    # 使用 seaborn.kdeplot 实现 Gaussian kernel density estimation [2]
    sns.kdeplot(x=df_gcl['Dim 1'], y=df_gcl['Dim 2'], 
                fill=True, 
                cmap="Blues", 
                levels=10,
                alpha=0.7)
    plt.title(f'GCLDM {item_or_user} Embeddings (KDE)', fontsize=14)
    plt.xlabel("Dim 1 (t-SNE)", fontsize=12)
    plt.ylabel("Dim 2 (t-SNE)", fontsize=12)
    plt.gca().set_aspect('equal', adjustable='box')


    # --- Plot Baseline (LightGCN) ---
    plt.subplot(1, 2, 2)
    sns.kdeplot(x=df_base['Dim 1'], y=df_base['Dim 2'], 
                fill=True, 
                cmap="Reds", 
                levels=10,
                alpha=0.7)
    plt.title(f'LightGCN Baseline {item_or_user} Embeddings (KDE)', fontsize=14)
    plt.xlabel("Dim 1 (t-SNE)", fontsize=12)
    plt.ylabel("Dim 2 (t-SNE)", fontsize=12)
    plt.gca().set_aspect('equal', adjustable='box')
    
    plt.suptitle(f'Embedding Distribution Visualization Comparison ({item_or_user})', fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    save_path = f'embedding_distribution_comparison_{item_or_user.lower()}.png'
    plt.savefig(save_path)
    print(f"\nVisualization saved to {save_path}")
    plt.show()

# ==============================================================================
# 3. 主程序执行 (Main Execution)
# ==============================================================================

if __name__ == "__main__":
    
    # --- 设定模型和路径 ---
    # 假设您的 GCLDM 和 LightGCN 模型检查点路径
    GCLDM_CKPT_PATH = "/root/ryw/Rec/RLMRec/encoder/checkpoint/gcldm/gcldm-amazon-2023.pth"
    LIGHTGCN_CKPT_PATH = "/root/ryw/Rec/RLMRec/encoder/checkpoint/lightgcn/lightgcn-amazon-2023.pth"
    
    # 采样数量
    N_SAMPLES = 5000

    print("--- 1. 加载 GCLDM 嵌入 ---")
    # 由于 GCLDM 包含了基于原型的对比学习 [1]，其嵌入预期分布更均匀 [2]
    gcldm_user_emb, gcldm_item_emb = load_trained_model_gcldm()

    print("\n--- 2. 加载 LightGCN 基线嵌入 ---")
    # LightGCN 作为基线模型，预期其嵌入分布可能集中在几个紧密集群 [2]
    lightgcn_user_emb, lightgcn_item_emb = load_trained_model_lightgcn()

    # ======================================================
    # 3. 物品嵌入可视化 (Item Embeddings Visualization)
    # NCL 论文主要展示了物品嵌入的均匀性 [2]
    # ======================================================
    
    print("\n--- 3. 降维和可视化物品嵌入 ---")
    df_gcl_item = tsne_reduction(gcldm_item_emb, num_samples=N_SAMPLES)
    df_base_item = tsne_reduction(lightgcn_item_emb, num_samples=N_SAMPLES)
    
    plot_kde_distribution(df_gcl_item, df_base_item, "Item")
    
    # ======================================================
    # 4. 用户嵌入可视化 (User Embeddings Visualization) (可选)
    # ======================================================

    print("\n--- 4. 降维和可视化用户嵌入 ---")
    df_gcl_user = tsne_reduction(gcldm_user_emb, num_samples=N_SAMPLES)
    df_base_user = tsne_reduction(lightgcn_user_emb, num_samples=N_SAMPLES)
    
    plot_kde_distribution(df_gcl_user, df_base_user, "User")