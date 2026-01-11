import pickle
import argparse
import os


def main():
    parser = argparse.ArgumentParser(description="Export all user profiles to txt (sorted by user_id)")
    parser.add_argument(
        "dataset",
        type=str,
        help="Dataset name, e.g., amazon / yelp / movielens"
    )
    args = parser.parse_args()
    dataset = args.dataset

    base_dir = "/root/ryw/Rec/RLMRec/data"
    pkl_path = os.path.join(base_dir, dataset, "usr_prf.pkl")
    save_path = os.path.join(base_dir, dataset, f"usr_prf_{dataset}.txt")

    if not os.path.exists(pkl_path):
        raise FileNotFoundError(f"Profile file not found: {pkl_path}")

    # 1. Load profiles
    with open(pkl_path, 'rb') as f:
        prf = pickle.load(f)

    # 2. Export all users, sorted by user_id
    with open(save_path, 'w', encoding='utf-8') as f:
        # ===== 文件头部元信息 =====
        f.write(f"# DATASET: {dataset}\n")
        f.write(f"# NUM_USERS: {len(prf)}\n")
        f.write("# FORMAT: USER_ID / PROFILE / REASONING\n\n")

        for uid in sorted(prf.keys()):
            info = prf[uid]
            f.write(f"USER_ID: {uid}\n")
            f.write("[PROFILE]\n")
            f.write(info["profile"].strip() + "\n")
            f.write("[REASONING]\n")
            f.write(info["reasoning"].strip() + "\n")
            f.write("-" * 40 + "\n\n")

    print(f"Saved {len(prf)} user profiles (sorted by user_id) to {save_path}")


if __name__ == "__main__":
    main()
