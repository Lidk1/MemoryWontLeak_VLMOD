"""
生成 VLMOD Task 2 提交文件（.txt），按官方评测格式：每行三个以空格分隔的整数（0/1/2）。

功能概述：
- 读取 `MonoMulti3D-ROPE/test` 下的 `*_obstacle.json` 文件；
- 从 `public_description` 解析文本约束（颜色/方位/对象名/数值范围等）；
- 对 `test_data` 每个对象逐一判定是否满足约束，输出三列布尔位（不足三列则补零，超过三列则截断）；
- 保存为同名 `.txt` 文件至 `submission_txt/` 目录。

用法：
- 直接执行本脚本：`python generate_submission_txt.py`；
- 生成的 `.txt` 文件可打包为 `submission_txt.zip` 并在评测网站提交。
"""

import os
import json
from typing import List, Dict, Any, Tuple

from libs import Object3D, extract_constraints, place_filter


def parse_test_line(line: str) -> Object3D:
    toks = line.strip().split()
    if len(toks) < 16:
        raise ValueError(f"Malformed test line (len={len(toks)}): {line}")
    raw: List[Any] = []
    raw.append(str(toks[0]))             # category
    raw.append(int(float(toks[1])))       # flag_a
    raw.append(int(float(toks[2])))       # flag_b
    raw.append(float(toks[3]))            # score_or_scale
    raw.append(float(toks[4]))            # x1
    raw.append(float(toks[5]))            # y1
    raw.append(float(toks[6]))            # x2
    raw.append(float(toks[7]))            # y2
    raw.append(float(toks[8]))            # dim_w
    raw.append(float(toks[9]))            # dim_h
    raw.append(float(toks[10]))           # dim_l
    raw.append(float(toks[11]))           # X
    raw.append(float(toks[12]))           # Y
    raw.append(float(toks[13]))           # Z
    raw.append(float(toks[14]))           # yaw
    raw.append(" ".join(toks[15:]))      # appearance (may contain hyphen)
    return Object3D(raw)


def load_test(json_path: str) -> Tuple[List[str], List[Object3D]]:
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    desc = data.get('public_description', [])
    if isinstance(desc, list):
        desc_list = [str(x) for x in desc]
    elif isinstance(desc, str):
        desc_list = [desc]
    else:
        desc_list = []
    lines = data.get('test_data', [])
    objs: List[Object3D] = []
    for line in lines:
        try:
            objs.append(parse_test_line(line))
        except Exception:
            continue
    return desc_list, objs


def object_matches_constraints(obj: Object3D, constraints: Dict[str, Any], img_wh: Tuple[float, float]) -> bool:
    # object and color must match if specified
    objects = constraints.get('objects', set())
    colors = constraints.get('colors', set())
    if objects and obj.category not in objects:
        return False
    if colors and obj.appearance not in colors:
        return False
    # place filter must pass if specified
    places = constraints.get('places', set())
    if places and not place_filter(obj, places, img_wh):
        return False
    # dims ranges: must be within specified ranges if provided
    dims_ranges = constraints.get('dims_ranges', {})
    def within(val: float, rng: Tuple[float, float]) -> bool:
        low, high = rng
        return (val >= low) and (val <= high)
    if 'length' in dims_ranges and not within(obj.dim_l, dims_ranges['length']):
        return False
    if 'width' in dims_ranges and not within(obj.dim_w, dims_ranges['width']):
        return False
    if 'height' in dims_ranges and not within(obj.dim_h, dims_ranges['height']):
        return False
    # distance range via Z
    dist_rng = constraints.get('distance_range', None)
    if dist_rng is not None:
        if not within(obj.Z, dist_rng):
            return False
    return True


def generate_for_file(json_path: str, out_dir: str) -> str:
    desc_list, objs = load_test(json_path)
    if not objs:
        # still write an empty file to follow naming requirement
        base = os.path.splitext(os.path.basename(json_path))[0]
        out_path = os.path.join(out_dir, f"{base}.txt")
        with open(out_path, 'w', encoding='utf-8') as f:
            pass
        return out_path

    # estimate image size from bbox spans (since images are not provided)
    max_x = max([o.x2 for o in objs])
    max_y = max([o.y2 for o in objs])
    img_wh = (max_x, max_y)

    # pre-parse constraints for each description
    constraints_list = [extract_constraints(d) for d in desc_list]

    lines: List[str] = []
    for o in objs:
        bits: List[str] = []
        for c in constraints_list:
            bits.append('1' if object_matches_constraints(o, c, img_wh) else '0')
        # ensure we output exactly three columns: pad/truncate
        if len(bits) < 3:
            bits = bits + ['0'] * (3 - len(bits))
        else:
            bits = bits[:3]
        lines.append(' '.join(bits))

    base = os.path.splitext(os.path.basename(json_path))[0]
    out_path = os.path.join(out_dir, f"{base}.txt")
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    return out_path


def main():
    test_dir = os.path.join('MonoMulti3D-ROPE', 'test')
    out_dir = os.path.join('.', 'submission_txt')
    os.makedirs(out_dir, exist_ok=True)

    files = [f for f in os.listdir(test_dir) if f.endswith('_obstacle.json')]
    files.sort()
    print(f"Found {len(files)} test JSONs in {test_dir}")
    for fname in files:
        fpath = os.path.join(test_dir, fname)
        out_path = generate_for_file(fpath, out_dir)
        print('Wrote', out_path)
    print('All txts written to', out_dir)


if __name__ == '__main__':
    main()