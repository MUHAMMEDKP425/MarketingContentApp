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
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    model = GPT2LMHeadModel.from_pretrained(model_name)
    model.resize_token_embeddings(len(tokenizer))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    return tokenizer, model, device

# Load resources
tokenizer, model, device = load_model()

# -----------------------------
# Helper functions
# -----------------------------
def safe_read_tabular(uploaded_file):
    try:
        uploaded_file.seek(0)
    except Exception:
        pass

    raw = uploaded_file.read()
    lower_name = getattr(uploaded_file, "name", "").lower()
    if lower_name.endswith((".xls", ".xlsx")):
        try:
            return pd.read_excel(io.BytesIO(raw))
        except Exception as e:
            raise ValueError(f"Could not read Excel file: {e}")

    encodings_to_try = ["utf-8", "utf-8-sig", "latin1", "cp1252", "iso-8859-1"]
    for enc in encodings_to_try:
        try:
            s = raw.decode(enc)
            return pd.read_csv(io.StringIO(s))
        except Exception:
            continue

    try:
        return pd.read_csv(io.StringIO(raw.decode("utf-8", errors="replace")), engine="python")
    except Exception as e:
        raise ValueError(
            "Failed to parse the uploaded file as CSV. Re-save as UTF-8 CSV or upload an xlsx."
        )


def create_full_prompt(product_name, tone="Friendly", keywords="", content_type="social post", audience="customers", features=""):
    examples = """
Example 1:
Product: Wireless Headphones
Tone: Friendly
Features: noise cancellation, 30-hour battery, bluetooth
Post: "🎧 Meet our new Wireless Headphones — crystal-clear sound, 30-hour battery life, and comfortable listening with noise cancellation. Grab yours today and enjoy a special launch discount!"

Example 2:
Product: Herbal Face Cream
Tone: Professional
Features: paraben-free, dermatologist tested, 24-hour hydration
Post: "Discover our dermatologist-tested Herbal Face Cream — paraben-free and clinically proven to hydrate for 24 hours. Restore your skin's natural glow. Order now for a limited-time discount."
"""

    base = f"Write a {tone} {content_type} to {audience} promoting a new {product_name}."
    if features:
        base += f" Key features: {features}."
    if keywords:
        base += f" Include keywords: {keywords}."
    base += " Keep it concise (1-2 short sentences), highlight benefits, and add a clear call-to-action."

    full_prompt = examples + f"\nNow write the post:\n{base}\nPost:"
    return full_prompt


def generate_content(prompt, max_length=100, num_variations=3, temperature=0.65, top_k=40, top_p=0.9):
    encoded = tokenizer(prompt, return_tensors="pt", truncation=True)
    input_ids = encoded.input_ids.to(device)

    # Ensure max_length does not exceed GPT-2's limit
    model_max_len = model.config.n_positions  # usually 1024
    if max_length + input_ids.shape[1] > model_max_len:
        max_length = model_max_len - input_ids.shape[1] - 1

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
        if "Post:" in text:
            text = text.split("Post:", 1)[-1].strip()
        sentences = [s.strip() for s in text.replace("\n", " ").split(". ") if s.strip()]
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

# Sidebar settings
with st.sidebar:
    st.header("Generation Settings")
    max_length = st.slider("Max length (tokens)", 60, 300, 120)
    num_variations = st.slider("Variations per prompt", 1, 5, 3)
    temperature = st.slider("Creativity (temperature)", 0.1, 1.5, 0.9)
    top_k = st.slider("Top-K (tokens)", 10, 200, 50)
    top_p = st.slider("Top-p (nucleus)", 0.5, 1.0, 0.95)

# Tabs
tab1, tab2 = st.tabs(["Single Product", "Batch Upload"])

with tab1:
    st.subheader("Single Product Generator")
    product_name = st.text_input("Product name")
    features = st.text_area("Key features (comma separated)")
    tone = st.selectbox("Tone", ["Friendly", "Professional", "Humorous", "Persuasive"], index=0)
    content_type = st.selectbox("Content type", ["social post", "ad copy", "product description", "email subject"], index=0)
    audience = st.text_input("Target audience", value="customers")
    keywords = st.text_input("Keywords")

    if st.button("Generate for single product"):
        if product_name:
            if not features.strip():
                features = "quality build, energy efficient, reliable performance"
            prompt = create_full_prompt(product_name, tone, keywords, content_type, audience, features)
            results = generate_content(prompt, max_length=max_length, num_variations=num_variations, temperature=temperature, top_k=top_k, top_p=top_p)
            df = pd.DataFrame([{"Product": product_name, "Tone": tone, "Variation": i+1, "Content": r} for i, r in enumerate(results)])
            st.dataframe(df)
            st.download_button("Download CSV", df.to_csv(index=False), file_name=f"{product_name.replace(' ','_')}_marketing.csv", mime="text/csv")
        else:
            st.error("Please enter a product name.")

with tab2:
    st.subheader("Batch generation from CSV")
    uploaded_file = st.file_uploader("Upload CSV or Excel (must have 'Product' column)", type=["csv","xls","xlsx"])

    if uploaded_file:
        try:
            df_products = safe_read_tabular(uploaded_file)
        except Exception as e:
            st.error(f"Error reading file: {e}")
        else:
            cols = [str(c).strip() for c in df_products.columns]
            col_map = {c.lower(): c for c in cols}
            if 'product' not in col_map:
                st.error("CSV must contain a 'Product' column (case-insensitive).")
            else:
                if st.button("Generate batch content"):
                    generated = []
                    for _, row in df_products.iterrows():
                        p = str(row[col_map['product']]).strip()
                        if not p:
                            continue
                        features_val = str(row[col_map.get('features','')] or "")
                        tone_val = str(row[col_map.get('tone','')] or "Friendly")
                        content_type_val = str(row[col_map.get('content_type', row.get('Content Type',''))] or "social post")
                        audience_val = str(row[col_map.get('audience','')] or "customers")
                        keywords_val = str(row[col_map.get('keywords','')] or "")
                        prompt = create_full_prompt(p, tone_val, keywords_val, content_type_val, audience_val, features_val)
                        variations = generate_content(prompt, max_length=max_length, num_variations=num_variations, temperature=temperature, top_k=top_k, top_p=top_p)
                        for i, v in enumerate(variations,1):
                            generated.append({"Product": p, "Tone": tone_val, "Variation": i, "Content": v})
                    df_out = pd.DataFrame(generated)
                    st.dataframe(df_out)
                    st.download_button("Download batch CSV", df_out.to_csv(index=False), file_name="batch_marketing_content.csv", mime="text/csv")

st.markdown("---")
st.markdown("Built with GPT-2 (transformers). For higher-quality outputs consider using OpenAI GPT-3.5/GPT-4 API.")
