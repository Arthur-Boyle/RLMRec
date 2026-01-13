import argparse
import torch
import numpy as np
import os

from config.configurator import configs
from data_utils.build_data_handler import build_data_handler
from models.bulid_model import build_model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--target_user", type=int, required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    # -------- configs（一次性生效）--------
    configs["model"]["name"] = args.model
    configs["data"]["name"] = args.dataset
    configs["device"] = args.device

    # -------- data --------
    data_handler = build_data_handler()
    data_handler.load_data()

    # -------- model --------
    model = build_model(data_handler).to(args.device)

    ckpt_path = (
        f"./encoder/checkpoint/{args.model}/"
        f"{args.model}-{args.dataset}-2023.pth"
    )
    print("[Load checkpoint]", ckpt_path)

    ckpt = torch.load(ckpt_path, map_location=args.device)
    model.load_state_dict(ckpt, strict=True)
    model.eval()

    n_users = model.user_num

    with torch.no_grad():
        _, _, gnn_embeds, *_ = model.forward()

    # Layer-0 user embeddings
    user_emb = gnn_embeds[0][:n_users].cpu().numpy()

    # -------- save --------
    os.makedirs("./encoder/tmp", exist_ok=True)
    out_path = f"./encoder/tmp/{args.model}_{args.dataset}_u{args.target_user}.npz"

    np.savez(out_path, user_emb=user_emb)
    print("[Saved]", out_path)


if __name__ == "__main__":
    main()
