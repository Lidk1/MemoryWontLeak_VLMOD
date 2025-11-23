import os
import json
from typing import List, Dict, Any, Tuple

from libs import Object3D, extract_constraints, load_text_scorer, rank_with_text_scorer


def parse_test_line(line: str) -> Object3D:
    """
    Parse a space-separated test_data line into Object3D according to expected indices:
    [class_name, idx_a, idx_b, score_or_scale, x1, y1, x2, y2, dim_w, dim_h, dim_l, X, Y, Z, yaw, appearance]
    """
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
    # appearance is the rest joined (handles hyphenated colors)
    raw.append(" ".join(toks[15:]))
    return Object3D(raw)


def load_test_objects(json_path: str) -> Tuple[str, List[Object3D]]:
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    # public_description may be list of sentences; join them
    desc = data.get('public_description', [])
    if isinstance(desc, list):
        query_text = ' '.join(desc)
    elif isinstance(desc, str):
        query_text = desc
    else:
        query_text = ''
    lines = data.get('test_data', [])
    objs: List[Object3D] = []
    for line in lines:
        try:
            objs.append(parse_test_line(line))
        except Exception:
            continue
    return query_text, objs


def main():
    test_dir = os.path.join('MonoMulti3D-ROPE', 'test')
    model_path = os.path.join('.', 'text_scorer.pt')
    scorer = load_text_scorer(model_path)

    results: List[Dict[str, Any]] = []
    top_scores: List[float] = []
    cat_hist: Dict[str, int] = {}

    files = [f for f in os.listdir(test_dir) if f.endswith('_obstacle.json')]
    files.sort()
    for fname in files:
        fpath = os.path.join(test_dir, fname)
        query_text, objs = load_test_objects(fpath)
        constraints = extract_constraints(query_text)

        ranked = rank_with_text_scorer(objs, constraints, scorer)
        matches = [dict(score=s, **o.as_output()) for s, o in ranked]
        top = matches[0] if matches else None
        if top is not None:
            top_scores.append(float(top['score']))
            cat_hist[top['category']] = cat_hist.get(top['category'], 0) + 1
        results.append({
            'file': fname,
            'query': query_text,
            'top': top,
            'top5': matches[:5]
        })

    summary = {
        'count': len(results),
        'mean_top_score': (sum(top_scores) / len(top_scores)) if top_scores else 0.0,
        'category_distribution': cat_hist,
        'results': results
    }

    out_path = os.path.join('.', 'batch_results_test.json')
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print('Saved batch results to', out_path)


if __name__ == '__main__':
    main()