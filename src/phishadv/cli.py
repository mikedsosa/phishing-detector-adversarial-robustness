"""Command-line demo: score one URL and show the smallest edit that flips the verdict.

    python -m phishadv train --dataset phreshphish --out models/phreshphish
    python -m phishadv score "http://example.com/login" --models-dir models/phreshphish

``score`` takes exactly one URL. There is deliberately no batch or file mode:
the tool is for explaining a detector's weak spots, not for mass-producing
evasive URLs. Every URL it prints is defanged (``hxxp``, ``[.]``). It never
resolves or visits a URL.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from .attacks import minimal_flip
from .data import PHRESH_CSV, RAW_CSV, SAMPLE_CSV, load_dataset, split
from .metrics import detection_metrics
from .models import load_detector, save_detector
from .report import defang

MODEL_NAMES = ("gbm_raw", "cnn_raw", "gbm_raw_at", "cnn_raw_at")
DATASETS = {"phreshphish": ("phreshphish", PHRESH_CSV), "phiusiil": ("phiusiil", RAW_CSV),
            "sample": ("phiusiil", SAMPLE_CSV)}


def explain(model, url: str, max_depth: int = 3, width: int = 64) -> dict:
    """Score ``url`` and search for the fewest-transformation edit that flips it (single URL only)."""
    if not isinstance(url, str) or "\n" in url:
        raise ValueError("explain() takes exactly one URL")
    return minimal_flip(model, url.strip(), max_depth=max_depth, width=width)


def format_result(res: dict, model_name: str, info: dict) -> str:
    verdict = "phishing" if res["flagged"] else "legitimate"
    lines = [f"URL:          {defang(res['url'])}",
             f"Model:        {model_name} (trained on {info.get('dataset', '?')}, seed {info.get('seed', '?')})",
             f"P(phishing):  {res['p_phish']:.3f}  -> classed as {verdict}"]
    if not res["flagged"]:
        lines.append("Nothing to flip: the model already lets this URL through.")
    elif res["found"]:
        n = len(res["steps"])
        lines.append(f"Smallest edit that flips the verdict: {n} transformation{'s' * (n != 1)} "
                     f"({res['queries']} model queries)")
        for i, st in enumerate(res["steps"], 1):
            lines.append(f"  {i}. {st['transformation']:<26} {defang(st['url'])}")
        lines.append(f"P(phishing) after: {res['p_after']:.3f}  -> would be classed as legitimate")
    else:
        lines.append(f"No edit of up to the search depth flips the verdict ({res['queries']} model queries).")
    return "\n".join(lines)


def cmd_score(args) -> int:
    path = Path(args.models_dir) / args.model
    if not (path / "meta.json").exists():
        print(f"No model at {path}. Train one first: python -m phishadv train --out {args.models_dir}",
              file=sys.stderr)
        return 2
    model, info = load_detector(path)
    res = explain(model, args.url, max_depth=args.max_depth)
    if args.json:
        safe = {**res, "url": defang(res["url"]),
                "adversarial": defang(res["adversarial"]) if res["adversarial"] else None,
                "steps": [{"transformation": s["transformation"], "url": defang(s["url"])} for s in res["steps"]]}
        print(json.dumps(safe, indent=2))
    else:
        print(format_result(res, args.model, info))
    return 0


def cmd_train(args) -> int:
    """Train the baseline (and optionally adversarially trained) GBM and Char-CNN and save them."""
    from .experiment import CONFIGS, adversarial_training, fit, make_model, seed_everything

    name, csv = DATASETS[args.dataset]
    if not Path(csv).exists():
        print(f"{csv} not found; run `make data` first", file=sys.stderr)
        return 2
    base = {"phreshphish": "phresh", "phiusiil": "full", "sample": "smoke"}[args.dataset]
    cfg = CONFIGS[base]
    seed_everything(args.seed)
    df = load_dataset(name, csv)
    sp = split(df, args.seed, max_train=cfg.max_train)
    out = Path(args.out)
    for kind in args.kinds:
        m = fit(make_model(kind, "raw", cfg), sp.train.url.tolist(), sp.train.y.to_numpy())
        variants = [(f"{kind}_raw", m)]
        if not args.no_adv_training:
            at, _ = adversarial_training(kind, "raw", m, sp.train, cfg)
            variants.append((f"{kind}_raw_at", at))
        for vname, model in variants:
            met = detection_metrics(sp.test.y.to_numpy(), model.predict_proba(sp.test.url.tolist()))
            save_detector(model, out / vname, meta={"dataset": name, "seed": args.seed, "clean_test": met})
            print(f"saved {out / vname}  test F1={met['f1']:.4f}")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="phishadv", description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("score", help="score ONE URL and show the smallest flipping edit (defanged)")
    s.add_argument("url", help="a single URL")
    s.add_argument("--model", choices=MODEL_NAMES, default="gbm_raw")
    s.add_argument("--models-dir", default="models/phreshphish")
    s.add_argument("--max-depth", type=int, default=3, choices=range(1, 5), metavar="{1..4}")
    s.add_argument("--json", action="store_true")
    s.set_defaults(func=cmd_score)
    t = sub.add_parser("train", help="train and save demo models")
    t.add_argument("--dataset", choices=sorted(DATASETS), default="phreshphish")
    t.add_argument("--out", default="models/phreshphish")
    t.add_argument("--seed", type=int, default=42)
    t.add_argument("--kinds", nargs="+", choices=("gbm", "cnn"), default=["gbm", "cnn"])
    t.add_argument("--no-adv-training", action="store_true")
    t.set_defaults(func=cmd_train)
    args = ap.parse_args(argv)
    np.set_printoptions(precision=3)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
