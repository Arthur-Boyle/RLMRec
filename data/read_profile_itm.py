import pickle
import argparse
import os


def main():
    parser = argparse.ArgumentParser(
        description="Export all item profiles to txt (sorted by item_id)"
    )
    parser.add_argument(
        "dataset",
        type=str,
        help="Dataset name, e.g., amazon / yelp / movielens"
    )
    args = parser.parse_args()
    dataset = args.dataset

    base_dir = "/root/ryw/Rec/RLMRec/data"
    pkl_path = os.path.join(base_dir, dataset, "itm_prf.pkl")
    save_path = os.path.join(base_dir, dataset, f"itm_prf_{dataset}.txt")

    if not os.path.exists(pkl_path):
        raise FileNotFoundError(f"Item profile file not found: {pkl_path}")

    # 1. Load item profiles
    with open(pkl_path, "rb") as f:
        prf = pickle.load(f)

    # 2. Export all items, sorted by item_id
    with open(save_path, "w", encoding="utf-8") as f:
        # ===== Header =====
        f.write(f"# DATASET: {dataset}\n")
        f.write(f"# NUM_ITEMS: {len(prf)}\n")
        f.write("# FORMAT: ITEM_ID / PROFILE / REASONING\n\n")

        # ===== Body =====
        for item_id in sorted(prf.keys()):
            info = prf[item_id]
            f.write(f"ITEM_ID: {item_id}\n")
            f.write("[PROFILE]\n")
            f.write(info["profile"].strip() + "\n")
            f.write("[REASONING]\n")
            f.write(info["reasoning"].strip() + "\n")
            f.write("-" * 40 + "\n\n")

    print(f"Saved {len(prf)} item profiles (sorted by item_id) to {save_path}")


if __name__ == "__main__":
    main()
