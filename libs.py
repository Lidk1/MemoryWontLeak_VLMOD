# -*- coding: utf-8 -*-
"""
VLMOD core library:
- Data parsing (Object3D, JSON loaders)
- Text constraint parsing (colors/places/objects/numeric ranges/relations)
- Rule-based matching and optional KAN scorer
- Supervised text-aware scorer (TextConstraintScorer) and training
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import os
import json
import re
import random
from ast import literal_eval
from typing import List, Dict, Any, Tuple, Optional
    
def reparameterize(mu, logvar):
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)
    return eps * std + mu

class KANLinear(torch.nn.Module):
    def __init__(
        self,
        in_features,
        out_features,
        grid_size=5,
        spline_order=3,
        scale_noise=0.1,
        scale_base=1.0,
        scale_spline=1.0,
        enable_standalone_scale_spline=True,
        base_activation=torch.nn.SiLU,
        grid_eps=0.02,
        grid_range=[-1, 1],
    ):
        super(KANLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.grid_size = grid_size
        self.spline_order = spline_order

        h = (grid_range[1] - grid_range[0]) / grid_size
        grid = (
            (
                torch.arange(-spline_order, grid_size + spline_order + 1) * h
                + grid_range[0]
            )
            .expand(in_features, -1)
            .contiguous()
        )
        self.register_buffer("grid", grid)

        self.base_weight = torch.nn.Parameter(torch.Tensor(out_features, in_features))
        self.spline_weight = torch.nn.Parameter(
            torch.Tensor(out_features, in_features, grid_size + spline_order)
        )
        if enable_standalone_scale_spline:
            self.spline_scaler = torch.nn.Parameter(
                torch.Tensor(out_features, in_features)
            )

        self.scale_noise = scale_noise
        self.scale_base = scale_base
        self.scale_spline = scale_spline
        self.enable_standalone_scale_spline = enable_standalone_scale_spline
        self.base_activation = base_activation()
        self.grid_eps = grid_eps

        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.kaiming_uniform_(self.base_weight, a=math.sqrt(5) * self.scale_base)
        with torch.no_grad():
            noise = (
                (
                    torch.rand(self.grid_size + 1, self.in_features, self.out_features)
                    - 1 / 2
                )
                * self.scale_noise
                / self.grid_size
            )
            self.spline_weight.data.copy_(
                (self.scale_spline if not self.enable_standalone_scale_spline else 1.0)
                * self.curve2coeff(
                    self.grid.T[self.spline_order : -self.spline_order],
                    noise,
                )
            )
            if self.enable_standalone_scale_spline:
                # torch.nn.init.constant_(self.spline_scaler, self.scale_spline)
                torch.nn.init.kaiming_uniform_(self.spline_scaler, a=math.sqrt(5) * self.scale_spline)

    def b_splines(self, x: torch.Tensor):
        """
        Compute the B-spline bases for the given input tensor.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_features).

        Returns:
            torch.Tensor: B-spline bases tensor of shape (batch_size, in_features, grid_size + spline_order).
        """
        assert x.dim() == 2 and x.size(1) == self.in_features

        grid: torch.Tensor = (
            self.grid
        )  # (in_features, grid_size + 2 * spline_order + 1)
        x = x.unsqueeze(-1)
        bases = ((x >= grid[:, :-1]) & (x < grid[:, 1:])).to(x.dtype)
        for k in range(1, self.spline_order + 1):
            bases = (
                (x - grid[:, : -(k + 1)])
                / (grid[:, k:-1] - grid[:, : -(k + 1)])
                * bases[:, :, :-1]
            ) + (
                (grid[:, k + 1 :] - x)
                / (grid[:, k + 1 :] - grid[:, 1:(-k)])
                * bases[:, :, 1:]
            )

        assert bases.size() == (
            x.size(0),
            self.in_features,
            self.grid_size + self.spline_order,
        )
        return bases.contiguous()

    def curve2coeff(self, x: torch.Tensor, y: torch.Tensor):
        """
        Compute the coefficients of the curve that interpolates the given points.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_features).
            y (torch.Tensor): Output tensor of shape (batch_size, in_features, out_features).

        Returns:
            torch.Tensor: Coefficients tensor of shape (out_features, in_features, grid_size + spline_order).
        """
        assert x.dim() == 2 and x.size(1) == self.in_features
        assert y.size() == (x.size(0), self.in_features, self.out_features)

        A = self.b_splines(x).transpose(
            0, 1
        )  # (in_features, batch_size, grid_size + spline_order)
        B = y.transpose(0, 1)  # (in_features, batch_size, out_features)
        solution = torch.linalg.lstsq(
            A, B
        ).solution  # (in_features, grid_size + spline_order, out_features)
        result = solution.permute(
            2, 0, 1
        )  # (out_features, in_features, grid_size + spline_order)

        assert result.size() == (
            self.out_features,
            self.in_features,
            self.grid_size + self.spline_order,
        )
        return result.contiguous()

    @property
    def scaled_spline_weight(self):
        return self.spline_weight * (
            self.spline_scaler.unsqueeze(-1)
            if self.enable_standalone_scale_spline
            else 1.0
        )

    def forward(self, x: torch.Tensor):
        assert x.size(-1) == self.in_features
        original_shape = x.shape
        x = x.reshape(-1, self.in_features)

        base_output = F.linear(self.base_activation(x), self.base_weight)
        spline_output = F.linear(
            self.b_splines(x).view(x.size(0), -1),
            self.scaled_spline_weight.view(self.out_features, -1),
        )
        output = base_output + spline_output
        
        output = output.reshape(*original_shape[:-1], self.out_features)
        return output

    @torch.no_grad()
    def update_grid(self, x: torch.Tensor, margin=0.01):
        assert x.dim() == 2 and x.size(1) == self.in_features
        batch = x.size(0)

        splines = self.b_splines(x)  # (batch, in, coeff)
        splines = splines.permute(1, 0, 2)  # (in, batch, coeff)
        orig_coeff = self.scaled_spline_weight  # (out, in, coeff)
        orig_coeff = orig_coeff.permute(1, 2, 0)  # (in, coeff, out)
        unreduced_spline_output = torch.bmm(splines, orig_coeff)  # (in, batch, out)
        unreduced_spline_output = unreduced_spline_output.permute(
            1, 0, 2
        )  # (batch, in, out)

        # sort each channel individually to collect data distribution
        x_sorted = torch.sort(x, dim=0)[0]
        grid_adaptive = x_sorted[
            torch.linspace(
                0, batch - 1, self.grid_size + 1, dtype=torch.int64, device=x.device
            )
        ]

        uniform_step = (x_sorted[-1] - x_sorted[0] + 2 * margin) / self.grid_size
        grid_uniform = (
            torch.arange(
                self.grid_size + 1, dtype=torch.float32, device=x.device
            ).unsqueeze(1)
            * uniform_step
            + x_sorted[0]
            - margin
        )

        grid = self.grid_eps * grid_uniform + (1 - self.grid_eps) * grid_adaptive
        grid = torch.concatenate(
            [
                grid[:1]
                - uniform_step
                * torch.arange(self.spline_order, 0, -1, device=x.device).unsqueeze(1),
                grid,
                grid[-1:]
                + uniform_step
                * torch.arange(1, self.spline_order + 1, device=x.device).unsqueeze(1),
            ],
            dim=0,
        )

        self.grid.copy_(grid.T)
        self.spline_weight.data.copy_(self.curve2coeff(x, unreduced_spline_output))

    def regularization_loss(self, regularize_activation=1.0, regularize_entropy=1.0):
        """
        Compute the regularization loss.

        This is a dumb simulation of the original L1 regularization as stated in the
        paper, since the original one requires computing absolutes and entropy from the
        expanded (batch, in_features, out_features) intermediate tensor, which is hidden
        behind the F.linear function if we want an memory efficient implementation.

        The L1 regularization is now computed as mean absolute value of the spline
        weights. The authors implementation also includes this term in addition to the
        sample-based regularization.
        """
        l1_fake = self.spline_weight.abs().mean(-1)
        regularization_loss_activation = l1_fake.sum()
        p = l1_fake / regularization_loss_activation
        regularization_loss_entropy = -torch.sum(p * p.log())
        return (
            regularize_activation * regularization_loss_activation
            + regularize_entropy * regularization_loss_entropy
        )

class KAN(torch.nn.Module):
    def __init__(
        self,
        layers_hidden,
        grid_size=5,
        spline_order=3,
        scale_noise=0.1,
        scale_base=1.0,
        scale_spline=1.0,
        base_activation=torch.nn.SiLU,
        grid_eps=0.02,
        grid_range=[-1, 1],
    ):
        super(KAN, self).__init__()
        self.grid_size = grid_size
        self.spline_order = spline_order

        self.layers = torch.nn.ModuleList()
        for in_features, out_features in zip(layers_hidden, layers_hidden[1:]):
            self.layers.append(
                KANLinear(
                    in_features,
                    out_features,
                    grid_size=grid_size,
                    spline_order=spline_order,
                    scale_noise=scale_noise,
                    scale_base=scale_base,
                    scale_spline=scale_spline,
                    base_activation=base_activation,
                    grid_eps=grid_eps,
                    grid_range=grid_range,
                )
            )

    def forward(self, x: torch.Tensor, update_grid=False):
        for layer in self.layers:
            if update_grid:
                layer.update_grid(x)
            x = layer(x)
        return x

    def regularization_loss(self, regularize_activation=1.0, regularize_entropy=1.0):
        return sum(
            layer.regularization_loss(regularize_activation, regularize_entropy)
            for layer in self.layers
        )
    
class LatentParams(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super(LatentParams, self).__init__()
        self.fc_mu = nn.Linear(input_dim, latent_dim)
        self.fc_logvar = nn.Linear(input_dim, latent_dim)
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            if m == self.fc_logvar:
                nn.init.constant_(m.weight, 0.0)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
            else:
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
    
    def forward(self, x):
        mu = self.fc_mu(x)         
        logvar = self.fc_logvar(x) 
        return mu, logvar

# -----------------------------
# Dataset parsing and grounding
# -----------------------------

class Object3D:
    def __init__(self, raw: List[Any]):
        # Expected format (length may vary, but indices below are consistent in provided JSONs):
        # [class_name, idx_a, idx_b, score_or_scale, x1, y1, x2, y2, dim_w, dim_h, dim_l, X, Y, Z, yaw, appearance]
        self.category: str = str(raw[0])
        self.flag_a: int = int(raw[1])
        self.flag_b: int = int(raw[2])
        self.score_or_scale: float = float(raw[3])
        self.x1: float = float(raw[4])
        self.y1: float = float(raw[5])
        self.x2: float = float(raw[6])
        self.y2: float = float(raw[7])
        self.dim_w: float = float(raw[8])
        self.dim_h: float = float(raw[9])
        self.dim_l: float = float(raw[10])
        self.X: float = float(raw[11])
        self.Y: float = float(raw[12])
        self.Z: float = float(raw[13])
        self.yaw: float = float(raw[14])
        self.appearance: str = str(raw[15]) if len(raw) > 15 else ""

    @property
    def bbox(self) -> Tuple[float, float, float, float]:
        return (self.x1, self.y1, self.x2, self.y2)

    @property
    def center(self) -> Tuple[float, float]:
        return ((self.x1 + self.x2) * 0.5, (self.y1 + self.y2) * 0.5)

    def as_output(self) -> Dict[str, Any]:
        return {
            "category": self.category,
            "3d_coords": [self.X, self.Y, self.Z],
            "3d_dims": [self.dim_w, self.dim_h, self.dim_l],
            "yaw": self.yaw,
            "bbox": [self.x1, self.y1, self.x2, self.y2],
            "appearance": self.appearance,
        }


def parse_label_3(label_list: List[str]) -> List[Object3D]:
    parsed: List[Object3D] = []
    for s in label_list:
        try:
            # Each entry is a string representation of a Python list
            arr = literal_eval(s)
            parsed.append(Object3D(arr))
        except Exception:
            # Skip malformed entries
            continue
    return parsed


def parse_calib(calib_str: str) -> Dict[str, Any]:
    vals = [float(x) for x in calib_str.split(',')]
    # Intrinsic-like 3x4 matrix stored row-major
    K = [vals[0:4], vals[4:8], vals[8:12]]
    return {"K": K}


def parse_denorm(denorm_str: str) -> List[float]:
    return [float(x) for x in denorm_str.split(',')]


def load_json_annotation(json_path: str) -> Dict[str, Any]:
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    anns_block = data[0] if len(data) > 0 else []
    meta_block = data[1][0] if len(data) > 1 and len(data[1]) > 0 else {}

    anns: List[Dict[str, Any]] = []
    for ann in anns_block:
        objs = parse_label_3(ann.get("label_3", []))
        anns.append({
            "ann_id": ann.get("ann_id"),
            "public_properties": ann.get("public_properties", []),
            "public_description": ann.get("public_description", ""),
            "objects": objs,
        })

    calib = parse_calib(meta_block.get("calib", "0,0,0,0,0,0,0,0,0,0,1,0"))
    denorm = parse_denorm(meta_block.get("denorm", "0,0,0,0"))
    return {"annotations": anns, "calib": calib, "denorm": denorm}


# -----------------------------
# Simple text constraint parser
# -----------------------------

_COLOR_MAP = {
    "black": "black", "white": "white", "red": "red", "silver-grey": "silver-grey",
    "silver": "silver-grey", "grey": "silver-grey", "gray": "silver-grey",
    "\u9ed1\u8272": "black", "\u767d\u8272": "white", "\u7ea2\u8272": "red", "\u94f6\u7070\u8272": "silver-grey", "\u7070\u8272": "silver-grey"
}

_PLACE_TOKENS = {
    "left": "left", "right": "right", "top": "top", "bottom": "bottom",
    "top left": "top-left", "top-right": "top-right", "bottom-left": "bottom-left", "bottom-right": "bottom-right",
    "\u5de6": "left", "\u53f3": "right", "\u4e0a": "top", "\u4e0b": "bottom", "\u5de6\u4e0a": "top-left", "\u53f3\u4e0a": "top-right", "\u5de6\u4e0b": "bottom-left", "\u53f3\u4e0b": "bottom-right"
}

_OBJECT_NAME_MAP = {
    "car": "car", "van": "van", "truck": "truck", "bus": "bus",
    "\u6c7d\u8f66": "car", "\u8d27\u8f66": "truck", "\u9762\u5305\u8f66": "van", "\u5df4\u58eb": "bus"
}


def extract_constraints(text: str) -> Dict[str, Any]:
    t = text.lower()
    constraints: Dict[str, Any] = {
        "colors": set(),
        "places": set(),
        "objects": set(),
        # numeric constraints
        "dims_ranges": {},  # keys: length|width|height -> (low, high)
        "distance_range": None,  # (low, high)
    }

    # Colors
    for k in _COLOR_MAP.keys():
        if k in t:
            constraints["colors"].add(_COLOR_MAP[k])

    # Places (multi-word first)
    for phrase in sorted(_PLACE_TOKENS.keys(), key=lambda x: -len(x)):
        if phrase in t:
            constraints["places"].add(_PLACE_TOKENS[phrase])

    # Object names
    for k in _OBJECT_NAME_MAP.keys():
        if k in t:
            constraints["objects"].add(_OBJECT_NAME_MAP[k])

    relations = []
    if ("left of" in t) or ("\u5de6\u4fa7" in t) or ("\u5de6\u8fb9" in t):
        relations.append("left-of")
    if ("right of" in t) or ("\u53f3\u4fa7" in t) or ("\u53f3\u8fb9" in t):
        relations.append("right-of")
    constraints["relations"] = relations

    # Numeric ranges: length/width/height (e.g., "length: 4.1 to 4.5 m", "4.1到4.5米长")
    range_patterns = [
        (r"(length|\u957f\u5ea6)\s*[:\uff1a]?\s*([0-9]+(?:\.[0-9]+)?)\s*(?:to|-|~|\u5230)\s*([0-9]+(?:\.[0-9]+)?)\s*(m|\u7c73)", "length"),
        (r"(width|\u5bbd\u5ea6)\s*[:\uff1a]?\s*([0-9]+(?:\.[0-9]+)?)\s*(?:to|-|~|\u5230)\s*([0-9]+(?:\.[0-9]+)?)\s*(m|\u7c73)", "width"),
        (r"(height|\u9ad8\u5ea6)\s*[:\uff1a]?\s*([0-9]+(?:\.[0-9]+)?)\s*(?:to|-|~|\u5230)\s*([0-9]+(?:\.[0-9]+)?)\s*(m|\u7c73)", "height"),
        (r"([0-9]+(?:\.[0-9]+)?)\s*(?:to|-|~|\u5230)\s*([0-9]+(?:\.[0-9]+)?)\s*(m|\u7c73)\s*(\u957f|\u5bbd|\u9ad8|long|wide|tall)?", None),
    ]
    for pat, key in range_patterns:
        for m in re.finditer(pat, t):
            if key is None:
                low = float(m.group(1)); high = float(m.group(2))
                qual = (m.group(4) or "").lower()
                qmap = {"\u957f": "length", "long": "length", "\u5bbd": "width", "wide": "width", "\u9ad8": "height", "tall": "height"}
                dim_key = qmap.get(qual)
                if dim_key:
                    constraints["dims_ranges"][dim_key] = (low, high)
            else:
                low = float(m.group(2)); high = float(m.group(3))
                constraints["dims_ranges"][key] = (low, high)

    # Distance: e.g., "distance: 107.4 m", "距离107米", "about 100 m away"
    dist_patterns = [
        r"distance\s*[:\uff1a]?\s*([0-9]+(?:\.[0-9]+)?)\s*m",
        r"\u8ddd\u79bb\s*([0-9]+(?:\.[0-9]+)?)\s*\u7c73",
        r"([0-9]+(?:\.[0-9]+)?)\s*m\s*(away|\u8fdc|\u5904)",
    ]
    for pat in dist_patterns:
        dm = re.search(pat, t)
        if dm:
            d = float(dm.group(1))
            tol = max(5.0, 0.1 * d)  # ±10% or ±5m
            constraints["distance_range"] = (d - tol, d + tol)
            break
    return constraints


def place_filter(obj: Object3D, places: set, img_wh: Tuple[float, float]) -> bool:
    if not places:
        return True
    cx, cy = obj.center
    W, H = img_wh
    flags = set()
    if cx < W * 0.33:
        flags.add("left")
    elif cx > W * 0.67:
        flags.add("right")
    else:
        flags.add("center")
    if cy < H * 0.33:
        flags.add("top")
    elif cy > H * 0.67:
        flags.add("bottom")
    else:
        flags.add("middle")

    # Combined quadrants
    if "top" in flags and "left" in flags:
        flags.add("top-left")
    if "top" in flags and "right" in flags:
        flags.add("top-right")
    if "bottom" in flags and "left" in flags:
        flags.add("bottom-left")
    if "bottom" in flags and "right" in flags:
        flags.add("bottom-right")

    return any(p in flags for p in places)


def match_objects(objects: List[Object3D], constraints: Dict[str, Any]) -> List[Object3D]:
    if not objects:
        return []
    # Estimate image size from bbox spans (since images are not provided)
    max_x = max([o.x2 for o in objects])
    max_y = max([o.y2 for o in objects])
    img_wh = (max_x, max_y)

    results_scored: List[Tuple[float, Object3D]] = []
    for o in objects:
        score = 0.0
        # Object name filter + reward
        if constraints["objects"]:
            if o.category not in constraints["objects"]:
                continue
            else:
                score += 1.0
        # Color filter + reward
        if constraints["colors"]:
            if o.appearance not in constraints["colors"]:
                continue
            else:
                score += 1.0
        # Place filter + soft reward
        if not place_filter(o, constraints["places"], img_wh):
            continue
        else:
            if constraints["places"]:
                # reward for being close to region center
                cx, cy = o.center
                W, H = img_wh
                region_targets = []
                for p in constraints["places"]:
                    if p == "left":
                        region_targets.append((W*0.165, cy))
                    elif p == "right":
                        region_targets.append((W*0.835, cy))
                    elif p == "top":
                        region_targets.append((cx, H*0.165))
                    elif p == "bottom":
                        region_targets.append((cx, H*0.835))
                    elif p == "top-left":
                        region_targets.append((W*0.165, H*0.165))
                    elif p == "top-right":
                        region_targets.append((W*0.835, H*0.165))
                    elif p == "bottom-left":
                        region_targets.append((W*0.165, H*0.835))
                    elif p == "bottom-right":
                        region_targets.append((W*0.835, H*0.835))
                    elif p == "center":
                        region_targets.append((W*0.5, H*0.5))
                if region_targets:
                    # normalize distance to [0,1] by dividing by diagonal
                    diag = (W**2 + H**2) ** 0.5 + 1e-6
                    dists = [(((cx - tx)**2 + (cy - ty)**2) ** 0.5) / diag for tx, ty in region_targets]
                    closeness = 1.0 - min(dists)
                    score += 0.5 * max(0.0, closeness)
        # Dimension ranges: soft reward by closeness to interval center
        dims_ranges = constraints.get("dims_ranges", {})
        if dims_ranges:
            def closeness_to_range(val: float, rng: Tuple[float, float]) -> float:
                low, high = rng
                mid = 0.5 * (low + high)
                half = max(1e-6, 0.5 * (high - low))
                return max(0.0, 1.0 - abs(val - mid) / half)
            if "length" in dims_ranges:
                score += 0.5 * closeness_to_range(o.dim_l, dims_ranges["length"])
            if "width" in dims_ranges:
                score += 0.5 * closeness_to_range(o.dim_w, dims_ranges["width"])
            if "height" in dims_ranges:
                score += 0.5 * closeness_to_range(o.dim_h, dims_ranges["height"])
        # Distance range via Z: soft reward by closeness to target
        dist_rng = constraints.get("distance_range", None)
        if dist_rng is not None:
            target = 0.5 * (dist_rng[0] + dist_rng[1])
            tol = max(1e-3, 0.5 * (dist_rng[1] - dist_rng[0]))
            score += 0.5 * max(0.0, 1.0 - abs(o.Z - target) / tol)

        results_scored.append((score, o))

    # Sort by score descending
    results_scored.sort(key=lambda p: -p[0])
    results = [p[1] for p in results_scored]

    # Relation reasoning (post-filter for left-of/right-of among ranked)
    if constraints.get("relations") and results:
        results.sort(key=lambda r: r.center[0])
        mid = max(1, len(results) // 2)
        if "left-of" in constraints["relations"]:
            results = results[:mid]
        if "right-of" in constraints["relations"]:
            results = results[-mid:]
    return results


def ground_from_json(json_path: str, query: str) -> Dict[str, Any]:
    bundle = load_json_annotation(json_path)
    anns = bundle["annotations"]
    all_objects: List[Object3D] = []
    for ann in anns:
        all_objects.extend(ann["objects"])  # pool all objects in the scene

    constraints = extract_constraints(query)
    matched = match_objects(all_objects, constraints)
    return {
        "json_path": json_path,
        "query": query,
        "matches": [m.as_output() for m in matched]
    }


def batch_ground(directory: str, query: str, limit: Optional[int] = None) -> List[Dict[str, Any]]:
    results: List[Dict[str, Any]] = []
    count = 0
    for fname in os.listdir(directory):
        if not fname.endswith("_obstacle.json"):
            continue
        fpath = os.path.join(directory, fname)
        results.append(ground_from_json(fpath, query))
        count += 1
        if limit and count >= limit:
            break
    return results


# -----------------------------
# Optional scoring with KAN
# -----------------------------

class AttributeScorer(nn.Module):
    def __init__(self, color_vocab: List[str]):
        super().__init__()
        self.color_to_idx = {c: i for i, c in enumerate(color_vocab)}
        # Attribute vector: [dims(3), xyz(3), center(2), color_onehot(len(color_vocab))]
        self.attr_dim = 3 + 3 + 2 + len(color_vocab)
        self.kan = KAN([self.attr_dim, 64, 1], grid_size=5, spline_order=3)

    def forward(self, objs: List[Object3D]) -> torch.Tensor:
        if not objs:
            return torch.empty(0)
        max_x = max([o.x2 for o in objs])
        max_y = max([o.y2 for o in objs])
        rows: List[List[float]] = []
        for o in objs:
            cx, cy = o.center
            color_vec = [0.0] * len(self.color_to_idx)
            if o.appearance in self.color_to_idx:
                color_vec[self.color_to_idx[o.appearance]] = 1.0
            rows.append([
                o.dim_w, o.dim_h, o.dim_l,
                o.X, o.Y, o.Z,
                cx / (max_x + 1e-6), cy / (max_y + 1e-6),
                *color_vec,
            ])
        x = torch.tensor(rows, dtype=torch.float32)
        score = self.kan(x)
        return score.squeeze(-1)


def rank_matches_with_kan(matches: List[Object3D]) -> List[Object3D]:
    scorer = AttributeScorer(["black", "white", "red", "silver-grey"])
    with torch.no_grad():
        scores = scorer(matches)
    # Sort by descending score
    pairs = list(zip(matches, scores.tolist()))
    pairs.sort(key=lambda p: -p[1])
    return [p[0] for p in pairs]


# -----------------------------
# Supervised text-aware scorer
# -----------------------------

class TextConstraintScorer(nn.Module):
    """
    A lightweight MLP that scores objects given text-derived constraints.
    Input features per object are constructed from closeness to numeric ranges,
    color/object match flags, and spatial region proximity. This directly models
    how well an object satisfies a query.
    """
    def __init__(self, in_dim: int = 8, hidden: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def _feature_closeness_to_range(val: float, rng: Tuple[float, float]) -> float:
    low, high = rng
    mid = 0.5 * (low + high)
    half = max(1e-6, 0.5 * (high - low))
    return max(0.0, 1.0 - abs(val - mid) / half)


def build_constraint_features(
    objs: List[Object3D],
    constraints: Dict[str, Any],
    img_wh: Tuple[float, float]
) -> torch.Tensor:
    """
    Build feature rows per object reflecting how each object matches the constraints.
    Feature order:
    [obj_match, color_match, place_closeness,
     len_close, wid_close, hei_close, dist_close, bbox_area_norm]
    """
    W, H = img_wh
    diag = (W**2 + H**2) ** 0.5 + 1e-6
    rows: List[List[float]] = []
    dims_ranges = constraints.get("dims_ranges", {})
    dist_rng = constraints.get("distance_range", None)
    objects = constraints.get("objects", set())
    colors = constraints.get("colors", set())
    places = constraints.get("places", set())

    # Precompute region centers
    region_targets = []
    for p in places:
        if p == "left":
            region_targets.append((W * 0.165, H * 0.5))
        elif p == "right":
            region_targets.append((W * 0.835, H * 0.5))
        elif p == "top":
            region_targets.append((W * 0.5, H * 0.165))
        elif p == "bottom":
            region_targets.append((W * 0.5, H * 0.835))
        elif p == "top-left":
            region_targets.append((W * 0.165, H * 0.165))
        elif p == "top-right":
            region_targets.append((W * 0.835, H * 0.165))
        elif p == "bottom-left":
            region_targets.append((W * 0.165, H * 0.835))
        elif p == "bottom-right":
            region_targets.append((W * 0.835, H * 0.835))
        elif p == "center":
            region_targets.append((W * 0.5, H * 0.5))

    for o in objs:
        cx, cy = o.center
        obj_match = 1.0 if (objects and (o.category in objects)) else 0.0
        color_match = 1.0 if (colors and (o.appearance in colors)) else 0.0
        # place closeness
        if region_targets:
            dists = [(((cx - tx)**2 + (cy - ty)**2) ** 0.5) / diag for tx, ty in region_targets]
            place_close = 1.0 - min(dists)
        else:
            place_close = 0.0
        # dims closeness
        len_close = _feature_closeness_to_range(o.dim_l, dims_ranges["length"]) if ("length" in dims_ranges) else 0.0
        wid_close = _feature_closeness_to_range(o.dim_w, dims_ranges["width"]) if ("width" in dims_ranges) else 0.0
        hei_close = _feature_closeness_to_range(o.dim_h, dims_ranges["height"]) if ("height" in dims_ranges) else 0.0
        # distance closeness via Z
        if dist_rng is not None:
            target = 0.5 * (dist_rng[0] + dist_rng[1])
            tol = max(1e-3, 0.5 * (dist_rng[1] - dist_rng[0]))
            dist_close = max(0.0, 1.0 - abs(o.Z - target) / tol)
        else:
            dist_close = 0.0
        # bbox area normalized
        area = max(1e-6, (o.x2 - o.x1) * (o.y2 - o.y1))
        bbox_area_norm = min(1.0, area / (W * H + 1e-6))

        rows.append([
            obj_match, color_match, place_close,
            len_close, wid_close, hei_close, dist_close, bbox_area_norm
        ])
    return torch.tensor(rows, dtype=torch.float32)


def _obj_key(o: Object3D) -> Tuple[Any, ...]:
    return (
        o.category,
        round(o.X, 3), round(o.Y, 3), round(o.Z, 3),
        round(o.dim_w, 3), round(o.dim_h, 3), round(o.dim_l, 3),
        round(o.yaw, 3), o.appearance,
        round(o.x1, 1), round(o.y1, 1), round(o.x2, 1), round(o.y2, 1),
    )


def _collect_scene_objects(anns: List[Dict[str, Any]]) -> List[Object3D]:
    seen = set()
    all_objs: List[Object3D] = []
    for ann in anns:
        for o in ann["objects"]:
            k = _obj_key(o)
            if k in seen:
                continue
            seen.add(k)
            all_objs.append(o)
    return all_objs


def train_text_scorer(
    json_root: str,
    epochs: int = 3,
    lr: float = 1e-3,
    neg_per_pos: int = 4,
    save_path: str = os.path.join(".", "text_scorer.pt"),
    file_limit: Optional[int] = None,
) -> str:
    """
    Supervised training using JSON annotations:
    - For each (image, ann), use ann["public_description"] to build constraints
      and score positives (label_3 parsed objects) higher than other objects
      in the same image.
    - Hinge ranking loss: sum(max(0, margin - s_pos + s_neg))
    Returns the path where the model is saved.
    """
    # Select device (prefer CUDA if available)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[train_text_scorer] Using device: {device}")
    model = TextConstraintScorer(in_dim=8, hidden=32).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    margin = 0.2

    files = [f for f in os.listdir(json_root) if f.endswith("_obstacle.json")]
    if file_limit:
        files = files[:file_limit]

    for ep in range(epochs):
        total_loss = 0.0
        count = 0
        random.shuffle(files)
        for fname in files:
            bundle = load_json_annotation(os.path.join(json_root, fname))
            anns = bundle["annotations"]
            # scene objects and image wh
            scene_objs = _collect_scene_objects(anns)
            if not scene_objs:
                continue
            W = max([o.x2 for o in scene_objs]); H = max([o.y2 for o in scene_objs])
            img_wh = (W, H)

            # Build a set for fast membership
            all_set = {_obj_key(o): o for o in scene_objs}

            for ann in anns:
                pos_objs = ann["objects"]
                if not pos_objs:
                    continue
                text_src = ann.get("public_description", "") or ann.get("text", "")
                constraints = extract_constraints(text_src)
                # negatives: those in scene not in positives
                pos_keys = {_obj_key(o) for o in pos_objs}
                neg_candidates = [o for k, o in all_set.items() if k not in pos_keys]
                if not neg_candidates:
                    continue
                # sample negatives
                random.shuffle(neg_candidates)
                neg_objs = neg_candidates[:max(1, min(neg_per_pos * len(pos_objs), len(neg_candidates)))]

                # features
                feats_pos = build_constraint_features(pos_objs, constraints, img_wh).to(device)
                feats_neg = build_constraint_features(neg_objs, constraints, img_wh).to(device)

                s_pos = model(feats_pos)
                s_neg = model(feats_neg)
                # pairwise hinge: each pos vs mean of neg
                s_neg_mean = s_neg.mean().expand_as(s_pos)
                loss = torch.clamp(margin - s_pos + s_neg_mean, min=0).mean()

                opt.zero_grad(); loss.backward(); opt.step()
                total_loss += loss.item(); count += 1

        print(f"[ep {ep+1}/{epochs}] loss={total_loss/max(1,count):.4f} ({count} steps)")

    torch.save(model.state_dict(), save_path)
    return save_path


def load_text_scorer(model_path: str, device: Optional[str] = None) -> TextConstraintScorer:
    # Choose target device: explicit, or auto-detect
    dev = torch.device(device if device else ("cuda" if torch.cuda.is_available() else "cpu"))
    model = TextConstraintScorer(in_dim=8, hidden=32)
    state = torch.load(model_path, map_location="cpu")
    model.load_state_dict(state)
    model = model.to(dev)
    model.eval()
    return model


def rank_with_text_scorer(
    objs: List[Object3D], constraints: Dict[str, Any], scorer: TextConstraintScorer
) -> List[Tuple[float, Object3D]]:
    if not objs:
        return []
    W = max([o.x2 for o in objs]); H = max([o.y2 for o in objs])
    # Ensure features are placed on the same device as the scorer
    feat_device = next(scorer.parameters()).device
    feats = build_constraint_features(objs, constraints, (W, H)).to(feat_device)
    with torch.no_grad():
        scores = scorer(feats).tolist()
    pairs = list(zip(scores, objs))
    pairs.sort(key=lambda p: -p[0])
    return pairs


def ground_from_json_precise(json_path: str, query: str, model_path: str) -> Dict[str, Any]:
    """
    Grounding using trained TextConstraintScorer. Returns ranked matches with scores.
    """
    bundle = load_json_annotation(json_path)
    anns = bundle["annotations"]
    scene_objs = _collect_scene_objects(anns)
    constraints = extract_constraints(query)
    scorer = load_text_scorer(model_path)
    ranked = rank_with_text_scorer(scene_objs, constraints, scorer)
    return {
        "json_path": json_path,
        "query": query,
        "matches": [dict(score=s, **o.as_output()) for s, o in ranked]
    }


def evaluate_text_scorer(
    json_root: str,
    model_path: str,
    file_limit: Optional[int] = None,
    topks: Tuple[int, int] = (1, 5),
) -> Dict[str, Any]:
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    scorer = load_text_scorer(model_path, device=str(dev))
    files = [f for f in os.listdir(json_root) if f.endswith("_obstacle.json")]
    if file_limit:
        files = files[:file_limit]
    total = 0
    hit1 = 0
    hit5 = 0
    for fname in files:
        bundle = load_json_annotation(os.path.join(json_root, fname))
        anns = bundle["annotations"]
        scene_objs = _collect_scene_objects(anns)
        if not scene_objs:
            continue
        all_set = {_obj_key(o): o for o in scene_objs}
        for ann in anns:
            pos_objs = ann["objects"]
            if not pos_objs:
                continue
            text_src = ann.get("public_description", "") or ann.get("text", "")
            constraints = extract_constraints(text_src)
            ranked = rank_with_text_scorer(scene_objs, constraints, scorer)
            pos_keys = {_obj_key(o) for o in pos_objs}
            total += 1
            if ranked:
                top1_key = _obj_key(ranked[0][1])
                if top1_key in pos_keys:
                    hit1 += 1
                k = min(topks[1], len(ranked))
                topk_keys = {_obj_key(ranked[i][1]) for i in range(k)}
                if topk_keys & pos_keys:
                    hit5 += 1
    acc1 = (hit1 / total) if total else 0.0
    acc5 = (hit5 / total) if total else 0.0
    return {"total": total, "top1_acc": acc1, "top5_acc": acc5}
