import argparse
import os


def get_args():

    parser = argparse.ArgumentParser()
    parser.add_argument("--method", type=str, default="DSTH")
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--hash-layer", type=str, default="linear")
    parser.add_argument("--save-dir", type=str, default="./result")
    parser.add_argument("--clip-path", type=str, default="./ViT-B-32.pt")
    parser.add_argument("--pretrained", type=str, default="None")
    parser.add_argument("--dataset", type=str, default="nuswide")
    parser.add_argument("--index-file", type=str, default="index.mat")
    parser.add_argument("--caption-file", type=str, default="caption.txt")
    parser.add_argument("--label-file", type=str, default="label.mat")
    parser.add_argument("--similarity-function", type=str, default="cosine")
    parser.add_argument("--loss-type", type=str, default="l2")
    parser.add_argument("--output-dim", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--max-words", type=int, default=32)
    parser.add_argument("--resolution", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--query-num", type=int, default=5000)
    parser.add_argument("--train-num", type=int, default=10000)
    parser.add_argument("--lr-decay-freq", type=int, default=5)
    parser.add_argument("--display-step", type=int, default=50)
    parser.add_argument("--seed", type=int, default=1814)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--lr-decay", type=float, default=0.9)
    parser.add_argument("--clip-lr", type=float, default=0.00001)
    parser.add_argument("--weight-decay", type=float, default=0.2)
    parser.add_argument("--warmup-proportion", type=float, default=0.1)
    parser.add_argument("--vartheta", type=float, default=0.5)
    parser.add_argument("--sim-threshold", type=float, default=0.1)
    parser.add_argument("--is-train", action="store_true", default='True')

    args = parser.parse_args()

    args.save_dir = os.path.join(args.save_dir, args.method, args.dataset, str(args.output_dim))
    os.makedirs(args.save_dir, exist_ok=True)
    return args
