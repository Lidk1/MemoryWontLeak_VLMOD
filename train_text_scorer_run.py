"""
训练文本约束评分器（TextConstraintScorer），用于提升基于文本约束的匹配排序质量。

数据：使用 `MonoMulti3D-ROPE/train/jsons` 目录下的标注 JSON；
输出：训练完成后保存模型权重至 `text_scorer.pt`。

用法：
- 直接执行本脚本：`python train_text_scorer_run.py`；
- 可调整 `epochs`, `lr`, `neg_per_pos`, `file_limit` 等参数。
"""

import os
from libs import train_text_scorer

def main():
    root = r"x:\VLMOD\MonoMulti3D-ROPE\train\jsons"
    print("Training TextConstraintScorer on:", root)
    save = train_text_scorer(
        root,
        epochs=10,  # 长轮次训练
        lr=1e-3,
        neg_per_pos=8,
        file_limit=None,  # 全量数据
        save_path=os.path.join('.', 'text_scorer.pt')
    )
    print("Model saved to:", save)

if __name__ == "__main__":
    main()