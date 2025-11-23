# VLMOD: Understanding Multi-Object World from Monocular View

> 本仓库为「2025 VLP 挑战赛参赛作品」。

Author: Keyu Guo, Yongle Huang, Shijie Sun, Xiangyu Song, Mingtao Feng, Zedong Liu, Huansheng Song, Tiantian Wang, Jianxin Li, Naveed Akhtar and Ajmal Saeed Mian



The paper has been accepted by **2025 IEEE Conference on Computer Vision and Pattern Recognition (CVPR2025)** 🎉.

<p align="center">

    <img src="https://i.imgur.com/waxVImv.png" alt="Oryx Video-ChatGPT">This repository provides **partial code** for the **VLMOD Challenge (Track B)** — *Understanding Multi-Object World from Monocular View*.  

Repository: https://github.com/Lidk1/MemoryWontLeak_VLMOD



![VLMOD.png](img/VLMOD.png)



The task focuses on **multi-object 3D Visual Grounding (3DVG)** based on **a single monocular RGB image**, enabling machines to interpret complex scenes and spatial relationships using natural language.



## 🧠 Task Description
Given a monocular RGB image and a complex language description (e.g., *"find the red cup on the left side of the table and the black keyboard on the right side"*),  
the goal is to predict **each referred object’s**:
- 3D position (x, y, z)
- 3D size (width, height, depth)
- Orientation (rotation angle)

## 🚧 Core Challenges
- Multi-object scene parsing  
- Spatial relationship modeling  
- Accurate 3D property estimation  

## 📂 Code Release
We have **open-sourced part of our implementation** to help the community explore and reproduce results.  
You are encouraged to:

- Reproduce and verify the released modules  
- Implement or improve other components  
- Contribute new ideas for monocular 3D visual grounding  

## 🚀 Quick Start (Baseline Grounding)

This repo includes a simple grounding baseline implemented in `libs.py` that parses the provided JSON annotations (MonoMulti3D-ROPE) and matches objects to a language query using rule-based constraints and an optional KAN-based ranking.

- Example (Python):

```python
from libs import ground_from_json, batch_ground

# Single file grounding
json_path = r"x:\MonoMulti-3DVG-main\MonoMulti3D-ROPE\train\jsons\1632_fa2sd4a11North151_420_1613710840_1613716786_1_obstacle.json"
query = "找出左上角的黑色汽车"
result = ground_from_json(json_path, query)
print(result)

# Optional: apply KAN-based ranking to matches
from libs import rank_matches_with_kan
result["matches"] = rank_matches_with_kan(result["matches"])

# Batch grounding on a directory (limit for preview)
dir_path = r"x:\MonoMulti-3DVG-main\MonoMulti3D-ROPE\train\jsons"
results = batch_ground(dir_path, query, limit=5)
for r in results:
    print(r["json_path"], len(r["matches"]))
```

- Output format per match:
  - `category`: object type (e.g., `car`)
  - `3d_coords`: `[X, Y, Z]`
  - `3d_dims`: `[width, height, length]`
  - `yaw`: rotation angle (radians)
  - `bbox`: `[x1, y1, x2, y2]`
  - `appearance`: color

Notes:
- Images are not included in this release; the baseline uses bounding box distribution in JSON to approximate spatial layout (e.g., left/right/top/bottom, quadrants).
- The baseline supports bilingual queries (English/Chinese) for colors, places, and simple relations.
- It also supports numeric constraints for dimensions and distance in queries:
  - Dimension ranges: "长度4.1到4.5米" / "length: 4.1 to 4.5 m"（支持 长/宽/高）
  - Distance: "距离约100米" / "distance: 100 m"（按Z轴过滤，容差±10%或±5m）
  - 示例：`"找出左侧黑色汽车，长度4到4.5米，距离约100米"`
- For full 3DVG training from RGB, integrate an image encoder and detector, then replace the rule-based matcher with multimodal attention and relation modeling.

## ⚙️ 环境配置（requirements）
- Python 3.10+
- 安装依赖：
  - 创建虚拟环境（可选）：`python -m venv .venv && .\.venv\Scripts\activate`
  - 安装：`pip install -r requirements.txt`
- 说明：`torch` 需根据本机 CUDA/CPU 环境选择合适的发行版；如需 GPU 训练，请参考 PyTorch 官网选择对应的版本与安装命令。

## 📄 开源许可与使用
- 许可证：本项目采用 `MIT License`（见仓库 `LICENSE` 文件）。
- 禁止使用限制性许可证（如 GPL 等），本仓库已符合赛事开源要求：公开可访问、包含环境配置（`requirements.txt`）、运行说明（README）。

## 🎯 更精确的模型（训练打分器）

为提升匹配精度，可以使用基于训练的 `TextConstraintScorer`。该打分器从标注 JSON 学习“数值范围、颜色、位置”等约束的权重，使排序更稳定、更贴合查询。

### 训练示例

```python
from libs import train_text_scorer

root = r"x:\MonoMulti-3DVG-main\MonoMulti3D-ROPE\train\jsons"
save = train_text_scorer(root, epochs=3, lr=1e-3, neg_per_pos=4, file_limit=200)
print("model saved to:", save)
```

- 数据来源：同一图像下，使用每条 `public_description` 的正样本（`label_3` 解析得到的对象）与同图像的其它对象构建排名任务。
- 损失函数：合页排名损失（hinge loss），推高正样本分数、压低负样本分数。
- 可选参数：
  - `epochs`（默认 3）：训练轮次。
  - `lr`（默认 1e-3）：学习率。
  - `neg_per_pos`（默认 4）：每个正样本配对的负样本数量。
  - `file_limit`：限制用于训练的文件数量（便于快速试跑）。
  - `save_path`：模型保存路径（默认 `./text_scorer.pt`）。

### 推理示例（精确排序）

```python
from libs import ground_from_json_precise

json_path = r"x:\MonoMulti-3DVG-main\MonoMulti3D-ROPE\train\jsons\1632_fa2sd4a11North151_420_1613710840_1613716786_10_obstacle.json"
query = "height : 1.4 to 1.8 m, appearance : white, distance : 59.9 m"
model_path = r"./text_scorer.pt"

res = ground_from_json_precise(json_path, query, model_path)
for m in res["matches"][:3]:  # 查看Top-3
    print(m["score"], m["category"], m["3d_coords"], m["3d_dims"], m["appearance"], m["bbox"]) 
```

- 返回中每项包含 `score`（分数越大越符合查询）以及基础版相同的对象字段（`category`, `3d_coords`, `3d_dims`, `yaw`, `bbox`, `appearance`）。
- 若需引入“朝向/遮挡”等更多约束，可在 `extract_constraints` 扩展解析，并在 `build_constraint_features` 补充特征。

## 📦 提交生成器（VLMOD Task 2）
本仓库包含一个用于官方评测提交的生成器：`generate_submission_txt.py`。

- 功能：为 `MonoMulti3D-ROPE/test` 下的每个 `*_obstacle.json` 生成同名 `.txt` 文件（位于 `submission_txt/`）。
- 输出格式：每行三列整数（空格分隔），分别对应 `public_description` 的前三条描述对该对象的匹配结果；当前版本为二值 `{0,1}`（未使用 `2`）。
- 打包：生成后可压缩为 `submission_txt.zip` 用于上传（见下）。

### 使用
```powershell
# 生成所有提交文件（Windows/PowerShell）
python generate_submission_txt.py

# 可选：打包为 zip 用于上传
Compress-Archive -Path submission_txt\* -DestinationPath submission_txt.zip
```

### 说明
- 读取字段：`public_description`（文本约束）、`test_data`（候选对象行）。
- 约束解析：`libs.extract_constraints` 与 `place_filter`，支持颜色、类别、方位、尺寸/距离区间等。
- 推理升级：训练完成后，推荐使用 `TextConstraintScorer` 的打分替换规则匹配，以提升指标。

数据说明：`MonoMulti3D-ROPE/test` 目录已包含评测 JSON；示例中的训练路径（如 `.../train/jsons`）为外部数据，需自行下载并调整为本地绝对路径。

## 🤝 Contribution
We welcome open discussions, reproduction efforts, and performance comparisons.  
Please feel free to submit issues or pull requests to share your work.

## 📜 License
This project is released for **academic and research purposes** only.



## **🏷️ Citation**

```bibtex
@inproceedings{guo2025beyond,
  title={Beyond Human Perception: Understanding Multi-Object World from Monocular View},
  author={Guo, Keyu and Huang, Yongle and Sun, Shijie and Song, Xiangyu and Feng, Mingtao and Liu, Zedong and Song, Huansheng and Wang, Tiantian and Li, Jianxin and Akhtar, Naveed and others},
  booktitle={Proceedings of the Computer Vision and Pattern Recognition Conference},
  pages={3751--3760},
  year={2025}
}
```

