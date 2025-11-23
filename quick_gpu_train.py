import os
import argparse
from libs import train_text_scorer, evaluate_text_scorer

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, default=r"x:\VLMOD\MonoMulti3D-ROPE\train\jsons")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--neg_per_pos", type=int, default=4)
    parser.add_argument("--file_limit", type=int, default=100)
    parser.add_argument("--save_path", type=str, default=os.path.join('.', 'text_scorer.pt'))
    parser.add_argument("--eval_file_limit", type=int, default=100)
    args = parser.parse_args()
    print(f"Quick train: epochs={args.epochs} files={args.file_limit}")
    save = train_text_scorer(
        args.root,
        epochs=args.epochs,
        lr=args.lr,
        neg_per_pos=args.neg_per_pos,
        file_limit=args.file_limit,
        save_path=args.save_path
    )
    print("Model saved to:", save)
    print("Evaluating on train annotations")
    metrics = evaluate_text_scorer(args.root, save, file_limit=args.eval_file_limit)
    print("Eval:", metrics)

if __name__ == "__main__":
    main()
