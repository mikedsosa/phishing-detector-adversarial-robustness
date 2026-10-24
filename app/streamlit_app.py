"""Local Streamlit demo: score one URL and show the smallest edit that flips the verdict.

    make models   # or: python -m phishadv train
    make app      # streamlit run app/streamlit_app.py  (binds to localhost only)

Single URL at a time, defanged output, no batch mode. Not meant to be hosted publicly.
"""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from phishadv.cli import MODEL_NAMES, explain  # noqa: E402
from phishadv.models import load_detector  # noqa: E402
from phishadv.report import defang  # noqa: E402

LABELS = {"gbm_raw": "Gradient boosting (baseline)", "cnn_raw": "Character CNN (baseline)",
          "gbm_raw_at": "Gradient boosting + adversarial training",
          "cnn_raw_at": "Character CNN + adversarial training"}


@st.cache_resource
def get_model(models_dir: str, name: str):
    return load_detector(Path(models_dir) / name)


st.set_page_config(page_title="Phishing URL detector: adversarial demo", layout="centered")
st.title("How few edits does it take to fool a phishing URL detector?")
st.caption("Research demo. Scores one URL offline; nothing is fetched or visited. All output is defanged.")

models_dir = st.sidebar.text_input("Models directory", "models/phreshphish")
available = [m for m in MODEL_NAMES if (Path(models_dir) / m / "meta.json").exists()]
if not available:
    st.error(f"No trained models in `{models_dir}`. Run `make models` first.")
    st.stop()
name = st.sidebar.selectbox("Model", available, format_func=lambda m: LABELS.get(m, m))
depth = st.sidebar.slider("Max edits to search", 1, 4, 3)
model, info = get_model(models_dir, name)
st.sidebar.write(f"Trained on **{info.get('dataset')}**, seed {info.get('seed')}.")

url = st.text_input("One URL", placeholder="http://login.example-account.xyz/verify.php?id=1")
if url:
    res = explain(model, url, max_depth=depth)
    st.metric("P(phishing)", f"{res['p_phish']:.3f}", help="Classed as phishing at 0.5 or above")
    st.code(defang(res["url"]), language=None)
    if not res["flagged"]:
        st.info("The model already classes this URL as legitimate.")
    elif res["found"]:
        st.warning(f"{len(res['steps'])} edit(s) flip the verdict (P = {res['p_after']:.3f}, "
                   f"{res['queries']} model queries).")
        for i, s in enumerate(res["steps"], 1):
            st.markdown(f"**{i}. {s['transformation']}**")
            st.code(defang(s["url"]), language=None)
    else:
        st.success(f"No combination of up to {depth} edits flips the verdict ({res['queries']} queries).")
