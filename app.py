import streamlit as st
from PIL import Image, ImageEnhance, ImageFilter
import pytesseract
import easyocr
import numpy as np
import io
import time

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Gallica OCR",
    page_icon="📜",
    layout="wide",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=IM+Fell+English:ital@0;1&family=JetBrains+Mono:wght@400;600&display=swap');

  html, body, [class*="css"] { font-family: 'IM Fell English', serif; }

  .stApp { background-color: #faf6ef; }

  h1 { font-family: 'IM Fell English', serif; font-size: 2.4rem !important; color: #2c1a0e; letter-spacing: 0.02em; }
  h2, h3 { font-family: 'IM Fell English', serif; color: #2c1a0e; }

  .result-box {
    background: #fffdf7;
    border: 1px solid #c8b89a;
    border-left: 4px solid #8b5e3c;
    border-radius: 4px;
    padding: 1.2rem 1.4rem;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.82rem;
    line-height: 1.7;
    color: #2c1a0e;
    white-space: pre-wrap;
    max-height: 500px;
    overflow-y: auto;
  }

  .engine-label {
    font-family: 'IM Fell English', serif;
    font-size: 1.1rem;
    font-weight: 600;
    color: #8b5e3c;
    margin-bottom: 0.4rem;
    display: flex;
    align-items: center;
    gap: 0.5rem;
  }

  .badge {
    display: inline-block;
    background: #8b5e3c;
    color: #faf6ef;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.7rem;
    padding: 2px 8px;
    border-radius: 2px;
    letter-spacing: 0.05em;
  }

  .metric-row {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.75rem;
    color: #9e8a6e;
    margin-top: 0.5rem;
  }

  .divider { border: none; border-top: 1px solid #d6c9b0; margin: 1.5rem 0; }

  .stButton > button {
    background-color: #2c1a0e;
    color: #faf6ef;
    font-family: 'IM Fell English', serif;
    font-size: 1rem;
    border: none;
    border-radius: 3px;
    padding: 0.5rem 1.6rem;
    cursor: pointer;
    transition: background 0.2s;
  }
  .stButton > button:hover { background-color: #8b5e3c; color: #faf6ef; }

  .stDownloadButton > button {
    background: transparent;
    border: 1px solid #8b5e3c;
    color: #8b5e3c;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.78rem;
    border-radius: 2px;
    padding: 0.3rem 1rem;
  }
  .stDownloadButton > button:hover { background: #8b5e3c; color: #faf6ef; }

  [data-testid="stSidebar"] { background-color: #f0e8d8; border-right: 1px solid #d6c9b0; }
  [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2,
  [data-testid="stSidebar"] h3, [data-testid="stSidebar"] p,
  [data-testid="stSidebar"] label { color: #2c1a0e; }
</style>
""", unsafe_allow_html=True)


# ── Image preprocessing ───────────────────────────────────────────────────────
def preprocess_image(img: Image.Image, contrast: float, sharpness: float, binarize: bool) -> Image.Image:
    img = img.convert("L")  # grayscale
    if contrast != 1.0:
        img = ImageEnhance.Contrast(img).enhance(contrast)
    if sharpness != 1.0:
        img = ImageEnhance.Sharpness(img).enhance(sharpness)
    if binarize:
        threshold = 140
        img = img.point(lambda p: 255 if p > threshold else 0, '1').convert("L")
    return img


# ── OCR engines ──────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Chargement d'EasyOCR…")
def load_easyocr(langs):
    return easyocr.Reader(langs, gpu=False)


def run_tesseract(img: Image.Image, lang: str) -> tuple[str, float]:
    start = time.time()
    config = "--oem 3 --psm 6"
    text = pytesseract.image_to_string(img, lang=lang, config=config)
    return text.strip(), round(time.time() - start, 2)


def run_easyocr(img: Image.Image, langs: list) -> tuple[str, float]:
    reader = load_easyocr(tuple(langs))
    arr = np.array(img)
    start = time.time()
    results = reader.readtext(arr, detail=0, paragraph=True)
    text = "\n".join(results)
    return text.strip(), round(time.time() - start, 2)


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Paramètres")
    st.markdown("---")

    st.markdown("**Moteurs actifs**")
    use_tesseract = st.checkbox("Tesseract", value=True)
    use_easyocr = st.checkbox("EasyOCR", value=True)

    st.markdown("---")
    st.markdown("**Langue du document**")
    lang_option = st.selectbox(
        "Langue principale",
        ["Français", "Latin", "Français + Latin", "Anglais"],
        index=0,
        label_visibility="collapsed"
    )
    lang_map = {
        "Français":          {"tess": "fra",      "easy": ["fr"]},
        "Latin":             {"tess": "lat",      "easy": ["la"]},
        "Français + Latin":  {"tess": "fra+lat",  "easy": ["fr", "la"]},
        "Anglais":           {"tess": "eng",      "easy": ["en"]},
    }
    langs = lang_map[lang_option]

    st.markdown("---")
    st.markdown("**Prétraitement de l'image**")
    contrast  = st.slider("Contraste",  0.5, 3.0, 1.4, 0.1)
    sharpness = st.slider("Netteté",    0.5, 3.0, 1.2, 0.1)
    binarize  = st.checkbox("Binarisation (noir/blanc dur)", value=False)

    st.markdown("---")
    st.markdown(
        "<small style='color:#9e8a6e;font-family:monospace'>Tesseract 5 · EasyOCR 1.7<br>"
        "Optimisé pour documents Gallica</small>",
        unsafe_allow_html=True
    )


# ── Main ──────────────────────────────────────────────────────────────────────
st.markdown("# 📜 Gallica OCR")
st.markdown(
    "<p style='color:#9e8a6e;font-size:1rem;margin-top:-0.8rem'>"
    "Comparateur de moteurs OCR pour documents anciens BnF/Gallica"
    "</p>",
    unsafe_allow_html=True
)
st.markdown("<hr class='divider'>", unsafe_allow_html=True)

uploaded = st.file_uploader(
    "Déposez votre image PNG (scan Gallica)",
    type=["png", "jpg", "jpeg", "tif", "tiff"],
    label_visibility="visible"
)

if uploaded:
    original = Image.open(uploaded)
    processed = preprocess_image(original, contrast, sharpness, binarize)

    col_img1, col_img2 = st.columns(2)
    with col_img1:
        st.markdown("**Image originale**")
        st.image(original, use_container_width=True)
    with col_img2:
        st.markdown("**Image prétraitée**")
        st.image(processed, use_container_width=True)

    st.markdown("<hr class='divider'>", unsafe_allow_html=True)

    if not use_tesseract and not use_easyocr:
        st.warning("Activez au moins un moteur dans le panneau latéral.")
        st.stop()

    if st.button("🔍 Lancer l'OCR"):
        results = {}

        if use_tesseract:
            with st.spinner("Tesseract en cours…"):
                try:
                    text, duration = run_tesseract(processed, langs["tess"])
                    results["Tesseract"] = {"text": text, "time": duration, "icon": "🏛️"}
                except Exception as e:
                    results["Tesseract"] = {"text": f"[Erreur] {e}", "time": 0, "icon": "🏛️"}

        if use_easyocr:
            with st.spinner("EasyOCR en cours (peut prendre 20–40s au premier lancement)…"):
                try:
                    text, duration = run_easyocr(processed, langs["easy"])
                    results["EasyOCR"] = {"text": text, "time": duration, "icon": "🔬"}
                except Exception as e:
                    results["EasyOCR"] = {"text": f"[Erreur] {e}", "time": 0, "icon": "🔬"}

        st.session_state["results"] = results

# ── Display results ───────────────────────────────────────────────────────────
if "results" in st.session_state and st.session_state["results"]:
    results = st.session_state["results"]
    engines = list(results.keys())

    cols = st.columns(len(engines))
    for col, engine in zip(cols, engines):
        r = results[engine]
        with col:
            st.markdown(
                f"<div class='engine-label'>{r['icon']} {engine} "
                f"<span class='badge'>{r['time']}s</span></div>",
                unsafe_allow_html=True
            )
            char_count = len(r["text"])
            word_count = len(r["text"].split())
            st.markdown(
                f"<div class='metric-row'>{word_count} mots · {char_count} caractères</div>",
                unsafe_allow_html=True
            )
            st.markdown(f"<div class='result-box'>{r['text'] or '<em>Aucun texte détecté</em>'}</div>",
                        unsafe_allow_html=True)

            if r["text"]:
                txt_bytes = r["text"].encode("utf-8")
                fname = f"{uploaded.name.rsplit('.', 1)[0]}_{engine.lower()}.txt"
                st.download_button(
                    label=f"⬇ Télécharger .txt ({engine})",
                    data=txt_bytes,
                    file_name=fname,
                    mime="text/plain",
                    key=f"dl_{engine}"
                )

    # Combined export
    if len(results) > 1:
        st.markdown("<hr class='divider'>", unsafe_allow_html=True)
        combined = "\n\n".join(
            f"{'='*40}\n{engine}\n{'='*40}\n{r['text']}"
            for engine, r in results.items()
        )
        st.download_button(
            label="⬇ Télécharger tous les résultats (.txt)",
            data=combined.encode("utf-8"),
            file_name=f"{uploaded.name.rsplit('.', 1)[0]}_ocr_compare.txt",
            mime="text/plain",
        )
