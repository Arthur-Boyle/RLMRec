import argparse
import torch
import numpy as np

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

    # ⚠️ 这里非常关键：
    # model / dataset 必须在程序启动时通过 configs 生效
    configs["model"]["name"] = args.model
    configs["data"]["name"] = args.dataset
    configs["device"] = args.device

    data_handler = build_data_handler()
    data_handler.load_data()

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

    user_emb = gnn_embeds[0][:n_users].cpu().numpy()

    # -------- graph distance --------
    u = args.target_user
    user_dist = data_handler.user_graph_dist[u]
    long_users = np.array(
        [v for v, d in user_dist.items() if d > 3],
        dtype=np.int32
    )

    out_path = (
        f"./encoder/tmp/"
        f"{args.model}_{args.dataset}_u{u}.npz"
    )
    print("[Save]", out_path)

    np.savez(
        out_path,
        user_emb=user_emb,
        long_users=long_users
    )


if __name__ == "__main__":
    main()
