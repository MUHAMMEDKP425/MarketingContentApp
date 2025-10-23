import streamlit as st
import pandas as pd
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch
import io

# -----------------------------
# Load model (cached)
# -----------------------------
@st.cache_resource
def load_model(model_name="gpt2"):
    """Load tokenizer and model. Uses CPU/GPU automatically if available."""
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    # Ensure tokenizer has a pad token (GPT2 doesn't by default)
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    model = GPT2LMHeadModel.from_pretrained(model_name)
    model.resize_token_embeddings(len(tokenizer))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    return tokenizer, model, device

# load resources
tokenizer, model, device = load_model()

# -----------------------------
# Helper: create fully-specified prompt
# -----------------------------
def safe_read_tabular(uploaded_file):
    """Try to read uploaded file as CSV (several encodings) or Excel.
    Returns a pandas DataFrame or raises a helpful error.
    """
    # Reset file pointer just in case
    try:
        uploaded_file.seek(0)
    except Exception:
        pass

    # Read raw bytes once
    raw = uploaded_file.read()
    # If it's an Excel file (xls/xlsx) try read_excel
    lower_name = getattr(uploaded_file, "name", "").lower()
    if lower_name.endswith((".xls", ".xlsx")):
        try:
            return pd.read_excel(io.BytesIO(raw))
        except Exception as e:
            raise ValueError(f"Could not read Excel file: {e}")

    # Try common encodings
    encodings_to_try = ["utf-8", "utf-8-sig", "latin1", "cp1252", "iso-8859-1"]
    for enc in encodings_to_try:
        try:
            s = raw.decode(enc)
            return pd.read_csv(io.StringIO(s))
        except Exception:
            # try next encoding
            pass

    # Try pandas CSV with engine='python' and guess delimiter
    try:
        return pd.read_csv(io.StringIO(raw.decode("utf-8", errors="replace")), engine="python")
    except Exception as e:
        # Give user a useful error
        raise ValueError(
            "Failed to parse the uploaded file as CSV. "
            "Common fixes: re-save as UTF-8 CSV from Excel, or upload an xlsx. "
            f"Underlying error: {e}"
        )


def create_full_prompt(product_name, tone="Friendly", keywords="", content_type="social post", audience="customers", features=""):
    # Few-shot examples to force marketing style
    examples = (
        "Example 1:
"
        "Product: Wireless Headphones
"
        "Tone: Friendly
"
        "Features: noise cancellation, 30-hour battery, bluetooth
"
        "Post: \"🎧 Meet our new Wireless Headphones — crystal-clear sound, 30-hour battery life, and comfortable listening with noise cancellation. Grab yours today and enjoy a special launch discount!\"

"
        "Example 2:
"
        "Product: Herbal Face Cream
"
        "Tone: Professional
"
        "Features: paraben-free, dermatologist tested, 24-hour hydration
"
        "Post: \"Discover our dermatologist-tested Herbal Face Cream — paraben-free and clinically proven to hydrate for 24 hours. Restore your skin's natural glow. Order now for a limited-time discount.\"

"
    )

    base = f"Write a {tone} {content_type} to {audience} promoting a new {product_name}."
    if features:
        base += f" Key features: {features}."
    if keywords:
        base += f" Include keywords: {keywords}."
    base += " Keep it concise (1-2 short sentences), highlight benefits, and add a clear call-to-action."

    # Put examples first so the model copies the style, then the real prompt
    full_prompt = examples + "Now write the post:
" + base + "
Post:"
    return full_prompt

# -----------------------------
# Generate multiple variations
# -----------------------------
def generate_content(prompt, max_length=100, num_variations=3, temperature=0.65, top_k=40, top_p=0.9):
    encoded = tokenizer(prompt, return_tensors="pt", truncation=True)
    input_ids = encoded.input_ids.to(device)

    outputs = model.generate(
        input_ids,
        max_length=max_length,
        num_return_sequences=num_variations,
        do_sample=True,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        no_repeat_ngram_size=3,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=getattr(tokenizer, 'eos_token_id', None)
    )

    variations = []
    for out in outputs:
        text = tokenizer.decode(out, skip_special_tokens=True)
        # remove everything up to the final "Post:" marker if present
        if "Post:" in text:
            text = text.split("Post:", 1)[-1].strip()
        # basic cleanup
        text = text.strip(' "
')
        # keep only first 2 sentences for concise posts
        sentences = [s.strip() for s in text.replace("
", " ").split(". ") if s.strip()]
        short = ". ".join(sentences[:2]).strip()
        if short and short[-1] not in ".!?":
            short += "."
        variations.append(short)
    return variations

# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(page_title="Marketing Content Generator", layout="wide")
st.title("Professional Marketing Content Generator")
st.caption("Generate marketing posts, descriptions, and ad copies with selectable tone, keywords, and batch export.")

# Layout: sidebar for settings
with st.sidebar:
    st.header("Generation Settings")
    default_tones = ["Friendly", "Professional", "Humorous", "Persuasive"]
    model_choice = st.selectbox("Model (local)", ["gpt2"], index=0)
    # generation params
    max_length = st.slider("Max length (tokens)", 60, 300, 120)
    num_variations = st.slider("Variations per prompt", 1, 5, 3)
    temperature = st.slider("Creativity (temperature)", 0.1, 1.5, 0.9)
    top_k = st.slider("Top-K (tokens)", 10, 200, 50)
    top_p = st.slider("Top-p (nucleus)", 0.5, 1.0, 0.95)

# Tabs for single vs batch
tab1, tab2 = st.tabs(["Single Product", "Batch Upload"])

with tab1:
    st.subheader("Single Product Generator")
    col1, col2 = st.columns([3, 1])
    with col1:
        product_name = st.text_input("Product name (e.g. 'Air Cooler')")
        features = st.text_area("Key features (comma separated) — e.g. 'energy efficient, quiet, remote control'")
        content_type = st.selectbox("Content type", ["social post", "ad copy", "product description", "email subject"], index=0)
        audience = st.text_input("Target audience (optional)", value="customers")
        keywords = st.text_input("Keywords (comma separated)")
        tones = st.multiselect("Select tone(s)", ["Friendly", "Professional", "Humorous", "Persuasive"], default=["Friendly"])
    with col2:
        st.write("
")
        if st.button("Generate for single product"):
            if not product_name:
                st.error("Please enter a product name.")
            else:
                if not features or not features.strip():
                    features = "quality build, energy efficient, reliable performance"
                all_results = []
                with st.spinner("Generating content..."):
                    for tone in tones:
                        prompt = create_full_prompt(product_name, tone, keywords, content_type, audience, features)
                        results = generate_content(prompt, max_length=max_length, num_variations=num_variations, temperature=temperature, top_k=top_k, top_p=top_p)
                        for i, r in enumerate(results, 1):
                            all_results.append({"Product": product_name, "Tone": tone, "Variation": i, "Content": r})
                df = pd.DataFrame(all_results)
                st.success("Content generated!")
                st.dataframe(df)
                csv = df.to_csv(index=False)
                st.download_button("Download results as CSV", csv, file_name=f"{product_name.replace(' ','_')}_marketing.csv", mime="text/csv")

            if not product_name:
                st.error("Please enter a product name.")
            else:
                all_results = []
                with st.spinner("Generating content..."):
                    for tone in tones:
                        prompt = create_full_prompt(product_name, tone, keywords, content_type, audience)
                        results = generate_content(prompt, max_length=max_length, num_variations=num_variations, temperature=temperature, top_k=top_k, top_p=top_p)
                        for i, r in enumerate(results, 1):
                            all_results.append({"Product": product_name, "Tone": tone, "Variation": i, "Content": r})
                df = pd.DataFrame(all_results)
                st.success("Content generated!")
                st.dataframe(df)
                csv = df.to_csv(index=False)
                st.download_button("Download results as CSV", csv, file_name=f"{product_name.replace(' ','_')}_marketing.csv", mime="text/csv")

with tab2:
    st.subheader("Batch generation from CSV")
    st.markdown("Upload a CSV with a column named **Product**. The app will generate content for each product.")
    uploaded_file = st.file_uploader("Upload CSV (column: 'Product')", type=["csv","xls","xlsx"])

    def safe_read_tabular(uploaded_file):
        import io
        # Reset file pointer
        try:
            uploaded_file.seek(0)
        except Exception:
            pass
        raw = uploaded_file.read()
        lower_name = getattr(uploaded_file, "name", "").lower()
        # Try Excel first
        if lower_name.endswith((".xls", ".xlsx")):
            try:
                return pd.read_excel(io.BytesIO(raw))
            except Exception as e:
                raise ValueError(f"Could not read Excel file: {e}")

        # Try common encodings for CSV
        encodings_to_try = ["utf-8", "utf-8-sig", "latin1", "cp1252", "iso-8859-1"]
        for enc in encodings_to_try:
            try:
                s = raw.decode(enc)
                return pd.read_csv(io.StringIO(s))
            except Exception:
                continue

        # Fallback: try pandas with python engine and replace errors
        try:
            return pd.read_csv(io.StringIO(raw.decode("utf-8", errors="replace")), engine="python")
        except Exception as e:
            raise ValueError(
                "Failed to parse the uploaded file as CSV. "
                "Common fixes: re-save as UTF-8 CSV from Excel, or upload an xlsx. "
                f"Underlying error: {e}"
            )

    if uploaded_file is not None:
        try:
            df_products = safe_read_tabular(uploaded_file)
        except Exception as e:
            st.error(f"Error reading file: {e}")
        else:
            # Normalize column names to detect fields in a case-insensitive way
            cols = [str(c).strip() for c in df_products.columns]
            col_map = {c.lower(): c for c in cols}
            def get_val(row, name):
                key = name.lower()
                if key in col_map:
                    return row[col_map[key]]
                return ""

            # Require at least a product column (case-insensitive)
            if 'product' not in col_map:
                st.error("CSV must contain a 'Product' column (case-insensitive).")
            else:
                st.info("Detected columns: " + ", ".join(cols))
                batch_run = st.button("Generate batch content")
                if batch_run:
                    generated = []
                    with st.spinner("Generating batch content..."):
                        for _, row in df_products.iterrows():
                            p = str(get_val(row, 'product')).strip()
                            if not p:
                                continue
                            # read optional per-row fields (case-insensitive) with sensible fallbacks
                            features_val = str(get_val(row, 'features') or "")
                            tone_val = str(get_val(row, 'tone') or "Friendly")
                            content_type_val = str(get_val(row, 'content_type') or get_val(row, 'content type') or "social post")
                            audience_val = str(get_val(row, 'audience') or "customers")
                            keywords_val = str(get_val(row, 'keywords') or get_val(row, 'keyword') or "")

                            prompt = create_full_prompt(p, tone_val, keywords_val, content_type_val, audience_val, features_val)
                            variations = generate_content(prompt, max_length=max_length, num_variations=num_variations, temperature=temperature, top_k=top_k, top_p=top_p)
                            for i, v in enumerate(variations, 1):
                                generated.append({"Product": p, "Tone": tone_val, "Variation": i, "Content": v})
                    df_out = pd.DataFrame(generated)
                    st.success("Batch content generated!")
                    st.dataframe(df_out)
                    st.download_button("Download batch CSV", df_out.to_csv(index=False), file_name="batch_marketing_content.csv", mime="text/csv")

# Footer
st.markdown("---")
st.markdown("Built with GPT-2 (transformers). For higher-quality outputs consider connecting an OpenAI GPT-3.5/GPT-4 API.")
