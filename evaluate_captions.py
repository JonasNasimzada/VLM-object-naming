#!/usr/bin/env python3
"""
Analyze VLM captions against ManyNames annotations:
- Count matches for top/incorrect responses
- Generate word clouds
- Plot domain distributions
"""

import argparse
import ast
import json
import re
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import pandas as pd
from wordcloud import WordCloud

IMG_ID_RE = re.compile(r"(\d+)")
ORIGINAL_IMG_RE = re.compile(r"^\d+\.jpg$")
TOKEN_RE = re.compile(r"\b[\w'-]+\b")


def tokenize(text):
    """Lowercase, simple word-tokenize, return a set for O(1) membership."""
    if not isinstance(text, str):
        return set()
    return set(m.group(0).lower() for m in TOKEN_RE.finditer(text))


def parse_mapping_cell(cell):
    """Cells are dict-likes; try JSON first, then Python literal. Returns {} on failure."""
    if pd.isna(cell):
        return {}
    s = str(cell).strip()
    if not s:
        return {}
    try:
        return json.loads(s)
    except Exception:
        try:
            return ast.literal_eval(s)
        except Exception:
            return {}


def load_vlm_captions(model, captions_dir):
    """Load a {image_name: caption} dict from captions/<model>.pkl."""
    captions_dir = Path(captions_dir)
    pkl_path = captions_dir / f"{model}.pkl"
    if not pkl_path.exists():
        raise FileNotFoundError(f"Missing captions file: {pkl_path}")
    data = pd.read_pickle(pkl_path)
    if not isinstance(data, dict):
        try:
            data = data.to_dict()
        except Exception:
            raise TypeError(f"Unsupported pickle format in {pkl_path}")
    out = {}
    for k, v in data.items():
        out[str(k)] = "" if v is None else str(v)
    return out


def split_original_cropped(captions):
    """Return (original_images, cropped_images) dicts."""
    originals = {k: v for k, v in captions.items() if ORIGINAL_IMG_RE.match(k)}
    cropped = {k: v for k, v in captions.items() if k.endswith("cropped.jpg")}
    return originals, cropped


def create_wordcloud(words, outfile):
    """Create and save a wordcloud if there are any words; no-op otherwise."""
    outfile = Path(outfile)
    words = list(words)
    if not words:
        return
    wc = WordCloud(width=800, height=400, background_color="white").generate(" ".join(words))
    fig = plt.figure(figsize=(10, 5))
    plt.imshow(wc, interpolation="bilinear")
    plt.axis("off")
    outfile.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outfile.as_posix(), bbox_inches="tight", dpi=150)
    plt.close(fig)


def plot_domain(counts, outfile):
    """Plot percentage bar chart for domain counts; skip if all zeros/empty."""
    outfile = Path(outfile)
    domains = list(counts.keys())
    values = list(counts.values())
    total = sum(values)
    if total <= 0:
        return
    percentages = [v / total for v in values]

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(domains, percentages)
    ax.set_xlabel("Domain")
    ax.set_ylabel("Percentage")
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    ax.bar_label(bars, labels=[f"{p * 100:.1f}%" for p in percentages], padding=3)

    outfile.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outfile.as_posix(), bbox_inches="tight", dpi=150)
    plt.close(fig)

def build_mn_index(df):
    """
    Build fast index: image_id -> {
        'domain': str,
        'top': set[str],
        'incorrect': set[str],
    }
    """
    index = {}
    for _, row in df.iterrows():
        try:
            img_id = int(row["vg_image_id"])
        except Exception:
            continue
        domain = str(row.get("domain", "unknown"))
        top_map = parse_mapping_cell(row.get("responses", ""))
        incorrect_map = parse_mapping_cell(row.get("incorrect", ""))
        index[img_id] = {
            "domain": domain,
            "top": set(map(str.lower, map(str, top_map.keys()))),
            "incorrect": set(map(str.lower, map(str, incorrect_map.keys()))),
        }
    return index


def match_words_in_caption(words, caption_tokens):
    """Return list of words present in the caption tokens (case-insensitive)."""
    if not words or not caption_tokens:
        return []
    return sorted(words.intersection(caption_tokens))


def analyze_model(model, captions, mn_index, plots_dir):
    model_plots_dir = Path(plots_dir) / model
    model_plots_dir.mkdir(parents=True, exist_ok=True)

    originals, cropped = split_original_cropped(captions)

    for variant_name, image_dict in (("original", originals), ("cropped", cropped)):
        top_hit_images = 0
        incorrect_hit_images = 0
        incorrect_wordcloud = []
        correct_wordcloud = []

        caption_tokens_cache = {k: tokenize(v) for k, v in image_dict.items()}

        for img_name, tokens in caption_tokens_cache.items():
            mid = IMG_ID_RE.search(img_name)
            if not mid:
                continue
            img_id = int(mid.group(1))
            entry = mn_index.get(img_id)
            if not entry:
                continue

            found_top = match_words_in_caption(entry["top"], tokens)
            found_incorrect = match_words_in_caption(entry["incorrect"], tokens)

            if found_top:
                top_hit_images += 1
                correct_wordcloud.extend(found_top)
            if found_incorrect:
                incorrect_hit_images += 1
                incorrect_wordcloud.extend(found_incorrect)

        print(f"VLM: {model}, {variant_name}, Top Responses: {top_hit_images}, Incorrect Responses: {incorrect_hit_images}")

        create_wordcloud(
            incorrect_wordcloud,
            model_plots_dir / f"{model}_{variant_name}_incorrect_wordcloud.png",
        )
        create_wordcloud(
            correct_wordcloud,
            model_plots_dir / f"{model}_{variant_name}_correct_wordcloud.png",
        )


def analyze_domains(model, captions, mn_index, plots_dir):
    model_plots_dir = Path(plots_dir) / model
    model_plots_dir.mkdir(parents=True, exist_ok=True)

    originals, cropped = split_original_cropped(captions)

    for variant_name, image_dict in (("original", originals), ("cropped", cropped)):
        domain_correct = defaultdict(int)
        domain_incorrect = defaultdict(int)

        caption_tokens_cache = {k: tokenize(v) for k, v in image_dict.items()}

        for img_name, tokens in caption_tokens_cache.items():
            mid = IMG_ID_RE.search(img_name)
            if not mid:
                continue
            img_id = int(mid.group(1))
            entry = mn_index.get(img_id)
            if not entry:
                continue
            domain = str(entry["domain"])
            if match_words_in_caption(entry["top"], tokens):
                domain_correct[domain] += 1
            if match_words_in_caption(entry["incorrect"], tokens):
                domain_incorrect[domain] += 1

        print(f"{model} {variant_name} domain_correct={dict(domain_correct)} domain_incorrect={dict(domain_incorrect)}\n")

        plot_domain(
            domain_correct,
            model_plots_dir / f"{model}_{variant_name}_correct_domain.png",
        )
        plot_domain(
            domain_incorrect,
            model_plots_dir / f"{model}_{variant_name}_incorrect_domain.png",
        )

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--vlm",
        nargs="+",
        default=["BLIP2", "llava", "git", "molmo"],
        help="VLM model names corresponding to captions/<model>.pkl (space-separated).",
    )
    parser.add_argument("--csv", type=str, default="manynames-en.tsv", help="ManyNames TSV file.")
    parser.add_argument("--captions-dir", type=str, default="captions", help="Directory containing <model>.pkl files.")
    parser.add_argument("--plots-dir", type=str, default="plots", help="Directory to write plots.")
    args = parser.parse_args()

    captions_dir = Path(args.captions_dir)
    plots_dir = Path(args.plots_dir)
    plots_dir.mkdir(parents=True, exist_ok=True)

    print(f"Selected VLM models: {args.vlm}")

    df = pd.read_csv(args.csv, encoding="utf-8", sep="\t")
    mn_index = build_mn_index(df)

    for model in args.vlm:
        captions = load_vlm_captions(model, captions_dir)
        analyze_model(model, captions, mn_index, plots_dir)
        analyze_domains(model, captions, mn_index, plots_dir)


if __name__ == "__main__":
    main()