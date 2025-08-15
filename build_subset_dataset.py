#!/usr/bin/env python3
"""
Build a completeness‑strict, ~30‑image stratified subset from your zero‑shot
caption outputs and create a dataset JSON for the localhost annotator.

Key guarantees & features:
- **Completeness enforced by default**: every selected image id has BOTH
  original and crop captions for **every** provided model.
- **Variable number of models**: pass any number of pickles via repeated
  --model flags, or use --models-glob to auto‑add all *.pkl in a folder.
- Copies the chosen image pairs into an output folder and emits a dataset JSON
  that the web app (index.html) can load.

Assumptions (tunable via CLI):
- Pickles map filename → caption (e.g., "123.jpg" and "123_cropped.jpf").
  The script also tries a few common alternative shapes; tweak iterate_entries()
  if your format differs.
- All actual images live in --images-dir (e.g., ./images).

Usage examples:
  # Explicit list of models
  python build_subset_dataset.py \
      --images-dir ./images \
      --model blip2=./pickles/blip2.pkl \
      --model show_and_tell=./pickles/show_and_tell.pkl \
      --model openflamingo=./pickles/openflamingo.pkl \
      --model git=./pickles/git.pkl \
      --model llava=./pickles/llava.pkl \
      --model molmo=./pickles/molmo.pkl \
      --out-dir subset_30 \
      --dataset-json dataset_subset.json \
      --n 30

  # Or auto‑discover: one model per *.pkl (model name = file stem)
  python build_subset_dataset.py \
      --images-dir ./images \
      --models-glob "./pickles/*.pkl" \
      --out-dir subset_30 \
      --dataset-json dataset_subset.json

Notes:
- If fewer than N ids satisfy completeness across all models, the script will
  sample as many as available and tell you how many it found.
- Use --allow-incomplete ONLY if you intentionally want to include items that
  aren’t covered by every model (not recommended for fair comparison).
"""

import argparse
import glob
import json
import os
import pickle
import random
import re
import shutil
from collections import defaultdict, Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Tuple, Any, List, Optional

# -------------------------------------------------------------
# Heuristic domain classification using keyword hits in captions
# -------------------------------------------------------------
DOMAIN_LEX: Dict[str, List[str]] = {
    # People / Clothing are split so we can see which dominates per image
    "people": [
        "person", "people", "man", "woman", "boy", "girl", "child", "kid", "face",
        "smile", "hands", "hair", "beard", "portrait"
    ],
    "clothing": [
        "shirt", "dress", "jeans", "jacket", "coat", "hat", "cap", "shoe", "sneaker",
        "boots", "scarf", "tie", "suit", "hoodie", "skirt"
    ],
    "home": [
        "sofa", "couch", "table", "chair", "lamp", "bed", "kitchen", "sink", "cup",
        "mug", "plate", "fork", "spoon", "pan", "pillow", "carpet"
    ],
    "buildings": [
        "building", "house", "church", "temple", "castle", "skyscraper", "bridge",
        "tower", "architecture", "apartment", "window", "door", "street"
    ],
    "food": [
        "food", "cake", "bread", "pizza", "burger", "sandwich", "pasta", "noodle",
        "salad", "soup", "rice", "fruit", "apple", "banana", "dessert", "cheese"
    ],
    "vehicles": [
        "car", "truck", "bus", "train", "motorcycle", "bike", "bicycle", "airplane",
        "boat", "ship", "scooter", "van"
    ],
    "animals_plants": [
        "dog", "cat", "bird", "horse", "cow", "sheep", "bear", "elephant", "giraffe",
        "zebra", "fish", "insect", "flower", "tree", "plant", "leaf"
    ],
    "sports_outdoor": [
        "ball", "soccer", "football", "basketball", "tennis", "baseball", "skate",
        "snow", "ski", "surf", "beach", "mountain", "hiking", "running", "bicycle"
    ],
}
ALL_DOMAINS = list(DOMAIN_LEX.keys()) + ["other"]

CROP_PAT = re.compile(r"(?P<id>.*?)(?:_(?:crop|cropped))?\.(?P<ext>[^.]+)$", re.IGNORECASE)

@dataclass
class FilePair:
    id: str
    orig_file: Optional[str] = None
    crop_file: Optional[str] = None
    captions_by_model: Dict[str, Dict[str, str]] = None  # {model: {"original": str, "crop": str}}

    def __post_init__(self):
        if self.captions_by_model is None:
            self.captions_by_model = {}

# -------------------------------------------------------------
# Utilities
# -------------------------------------------------------------

def base_id_from_filename(name: str) -> Tuple[str, bool]:
    """Return (base_id, is_crop) stripped of extension and _crop/_cropped suffix.
    Examples:
      123.jpg -> ("123", False)
      123_cropped.jpf -> ("123", True)
      path/to/ABC_crop.PNG -> ("ABC", True)
    """
    n = os.path.basename(name)
    m = CROP_PAT.match(n)
    if not m:
        root, _ = os.path.splitext(n)
        return root, False
    base = m.group("id")
    is_crop = bool(re.search(r"_(?:crop|cropped)(?=\.[^.]+$)", n, re.IGNORECASE))
    return base, is_crop


def iterate_entries(obj: Any) -> Iterable[Tuple[str, str]]:
    """Iterate (filename, caption) pairs from common pickle shapes."""
    if isinstance(obj, dict):
        for k, v in obj.items():
            if isinstance(v, (str, bytes)):
                cap = v.decode("utf-8", "ignore") if isinstance(v, bytes) else str(v)
                yield str(k), cap
            elif isinstance(v, dict):
                # Try a few common keys
                cand = v.get("caption") or v.get("text") or v.get("pred")
                if isinstance(cand, (str, bytes)):
                    cap = cand.decode("utf-8", "ignore") if isinstance(cand, bytes) else str(cand)
                    yield str(k), cap
    elif isinstance(obj, list):
        for e in obj:
            if isinstance(e, tuple) and len(e) >= 2:
                fn, cap = e[0], e[1]
                if isinstance(cap, (bytes, bytearray)):
                    cap = cap.decode("utf-8", "ignore")
                yield str(fn), str(cap)
            elif isinstance(e, dict):
                fn = e.get("filename") or e.get("file") or e.get("image")
                cap = e.get("caption") or e.get("text") or e.get("pred")
                if fn and cap:
                    if isinstance(cap, (bytes, bytearray)):
                        cap = cap.decode("utf-8", "ignore")
                    yield str(fn), str(cap)


def classify_domain(texts: Iterable[str]) -> str:
    counts = Counter()
    blob = (" ".join(texts)).lower()
    for dom, keys in DOMAIN_LEX.items():
        for k in keys:
            if f" {k} " in f" {blob} ":
                counts[dom] += 1
    if not counts:
        return "other"
    return counts.most_common(1)[0][0]


def allocate_samples(grouped: Dict[str, List[str]], n: int, seed: int = 42) -> List[str]:
    """Evenly draw up to n ids across present groups with spillover handling."""
    rng = random.Random(seed)
    present = {k: v[:] for k, v in grouped.items() if v}
    for v in present.values():
        rng.shuffle(v)
    domains = list(present.keys())
    if not domains:
        return []
    per = max(1, n // len(domains))

    chosen: List[str] = []
    # First pass: per domain
    for d in domains:
        take = min(per, len(present[d]))
        chosen += present[d][:take]
        present[d] = present[d][take:]
    # Remainder: rotate domains and take one each while available
    i = 0
    while len(chosen) < n and any(present.values()):
        d = domains[i % len(domains)]
        if present[d]:
            chosen.append(present[d].pop(0))
        i += 1
    return chosen[:n]

# -------------------------------------------------------------
# Main pipeline
# -------------------------------------------------------------

def collect_models(args) -> List[Tuple[str, Path]]:
    models: List[Tuple[str, Path]] = []
    # From explicit --model flags
    if args.model:
        for spec in args.model:
            if "=" not in spec:
                raise SystemExit(f"--model must be name=path.pkl, got: {spec}")
            name, path = spec.split("=", 1)
            name = name.strip()
            p = Path(path.strip())
            if not p.exists():
                raise SystemExit(f"Pickle not found: {p}")
            models.append((name, p))
    # From glob
    if args.models_glob:
        for fn in sorted(glob.glob(args.models_glob)):
            p = Path(fn)
            if p.is_file() and p.suffix.lower() == ".pkl":
                name = p.stem
                if all(name != m[0] for m in models):
                    models.append((name, p))
    if not models:
        raise SystemExit("No models provided. Use --model name=path.pkl and/or --models-glob.")
    return models


def main():
    ap = argparse.ArgumentParser(description="Build a completeness‑strict subset and dataset JSON for the annotator.")
    ap.add_argument("--images-dir", required=True, type=Path, help="Folder containing all images (originals and crops)")
    ap.add_argument("--model", action="append",
                    help="Repeat: name=path_to_pickle.pkl (e.g., blip2=./pickles/blip2.pkl)")
    ap.add_argument("--models-glob", type=str, default=None,
                    help="Optional glob to auto‑add models (name = file stem), e.g. './pickles/*.pkl'")
    ap.add_argument("--out-dir", type=Path, default=Path("person_eval/subset_30"), help="Output folder for the subset images")
    ap.add_argument("--dataset-json", type=Path, default=Path("dataset_subset.json"), help="Path to write dataset JSON")
    ap.add_argument("--n", type=int, default=30, help="Target number of image ids to sample")
    ap.add_argument("--seed", type=int, default=42, help="Random seed for sampling")
    ap.add_argument("--allow-incomplete", action="store_true",
                    help="Allow items that lack orig+crop for some models (NOT recommended)")
    args = ap.parse_args()

    images_dir: Path = args.images_dir
    out_dir: Path = args.out_dir
    out_images: Path = out_dir / "images"

    models = collect_models(args)
    model_names = [m[0] for m in models]

    # Load pickles and merge
    @dataclass
    class Accum:
        pairs: Dict[str, FilePair]
        coverage: Dict[str, Dict[str, int]]  # model -> {"orig":count, "crop":count, "both":count}

    acc = Accum(pairs={}, coverage=defaultdict(lambda: {"orig":0, "crop":0, "both":0}))

    for model_name, pkl_path in models:
        with open(pkl_path, "rb") as f:
            data = pickle.load(f)
        seen_orig = set(); seen_crop = set()
        for filename, caption in iterate_entries(data):
            base_id, is_crop = base_id_from_filename(filename)
            fp = acc.pairs.get(base_id)
            if fp is None:
                fp = FilePair(id=base_id)
                acc.pairs[base_id] = fp
            view = "crop" if is_crop else "original"
            fp.captions_by_model.setdefault(model_name, {})[view] = caption
            if view == "original":
                seen_orig.add(base_id)
                if fp.orig_file is None:
                    fp.orig_file = os.path.basename(filename)
            else:
                seen_crop.add(base_id)
                if fp.crop_file is None:
                    fp.crop_file = os.path.basename(filename)
        # coverage tallies
        acc.coverage[model_name]["orig"] = len(seen_orig)
        acc.coverage[model_name]["crop"] = len(seen_crop)
        acc.coverage[model_name]["both"] = len(seen_orig & seen_crop)

    # Completeness filtering (default strict)
    def has_both_views_for_all_models(fp: FilePair) -> bool:
        for m in model_names:
            views = fp.captions_by_model.get(m, {})
            if "original" not in views or "crop" not in views:
                return False
        return fp.orig_file is not None and fp.crop_file is not None

    def has_images(fp: FilePair) -> bool:
        return (fp.orig_file and (images_dir / fp.orig_file).exists() and
                fp.crop_file and (images_dir / fp.crop_file).exists())

    all_ids = list(acc.pairs.keys())
    if args.allow_incomplete:
        def ok(fp: FilePair) -> bool:
            # at least one model provides both views and both image files exist
            return has_images(fp) and any(("original" in v and "crop" in v) for v in fp.captions_by_model.values())
        kept_ids = [i for i in all_ids if ok(acc.pairs[i])]
    else:
        kept_ids = [i for i in all_ids if has_both_views_for_all_models(acc.pairs[i]) and has_images(acc.pairs[i])]

    if not kept_ids:
        print("\nCoverage report (ids with both captions per model):")
        for m in model_names:
            c = acc.coverage[m]
            print(f"  {m:16s} orig={c['orig']:5d} crop={c['crop']:5d} both={c['both']:5d}")
        raise SystemExit("\nNo items satisfy the completeness criteria across all models.\nCheck your pickles and --images-dir, or use --allow-incomplete.")

    # Domain classification per id (use all captions across models/views for that id)
    id_domain: Dict[str, str] = {}
    for i in kept_ids:
        fp = acc.pairs[i]
        texts: List[str] = []
        for m in model_names:
            views = fp.captions_by_model.get(m, {})
            if "original" in views: texts.append(views["original"])
            if "crop" in views: texts.append(views["crop"])
        dom = classify_domain(texts)
        id_domain[i] = dom

    # Group and sample
    grouped: Dict[str, List[str]] = defaultdict(list)
    for i, dom in id_domain.items():
        grouped[dom].append(i)

    target_n = args.n
    if len(kept_ids) < target_n:
        print(f"\nOnly {len(kept_ids)} ids satisfy completeness; sampling that many instead of {target_n}.")
        target_n = len(kept_ids)

    chosen_ids = allocate_samples(grouped, target_n, seed=args.seed)
    if not chosen_ids:
        raise SystemExit("Sampling failed (no domains present after filtering).")

    # Prepare output folder and copy images
    out_images.mkdir(parents=True, exist_ok=True)

    for i in chosen_ids:
        fp = acc.pairs[i]
        src_o = images_dir / fp.orig_file
        src_c = images_dir / fp.crop_file
        shutil.copy2(src_o, out_images / fp.orig_file)
        shutil.copy2(src_c, out_images / fp.crop_file)

    # Build dataset JSON
    rel_prefix = os.path.relpath(out_images, args.dataset_json.parent)

    items = []
    for i in chosen_ids:
        fp = acc.pairs[i]
        captions = {}
        for m in model_names:
            views = fp.captions_by_model.get(m, {})
            captions[m] = {
                "original": views.get("original", ""),
                "crop": views.get("crop", ""),
            }
        items.append({
            "id": fp.id,
            "image": str(Path(rel_prefix) / fp.orig_file).replace("\\", "/"),
            "crop": str(Path(rel_prefix) / fp.crop_file).replace("\\", "/"),
            "captions": captions,
            "domain": id_domain.get(i, "other"),
        })

    dataset = {
        "name": f"subset_{len(chosen_ids)}",
        "models": model_names,
        "items": items,
    }

    args.dataset_json.parent.mkdir(parents=True, exist_ok=True)
    with open(args.dataset_json, "w", encoding="utf-8") as f:
        json.dump(dataset, f, ensure_ascii=False, indent=2)

    # Coverage + summary
    print("\nCoverage report (ids with both captions per model in ALL data):")
    for m in model_names:
        c = acc.coverage[m]
        print(f"  {m:16s} orig={c['orig']:5d} crop={c['crop']:5d} both={c['both']:5d}")

    dom_counts_all = Counter(id_domain[i] for i in kept_ids)
    dom_counts_chosen = Counter(id_domain[i] for i in chosen_ids)

    print("\nDone. Subset created:\n")
    print(f"  Images copied to: {out_images}")
    print(f"  Dataset JSON:     {args.dataset_json}")
    print(f"  Models:           {', '.join(model_names)}")
    print("\nDomains in pool after filtering:")
    for d in sorted(dom_counts_all, key=lambda k: (-dom_counts_all[k], k)):
        print(f"  {d:15s} : {dom_counts_all[d]}")
    print("\nDomains in CHOSEN subset:")
    for d in sorted(dom_counts_chosen, key=lambda k: (-dom_counts_chosen[k], k)):
        print(f"  {d:15s} : {dom_counts_chosen[d]}")
    print("\nLoad the JSON in your annotator (index.html) and you can start comparing.")


if __name__ == "__main__":
    main()
