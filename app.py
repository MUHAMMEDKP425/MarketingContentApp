import streamlit as st
import pandas as pd
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch

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
def create_full_prompt(product_name, tone="Friendly", keywords="", content_type="social post", audience="customers"):
    """Automatically expands short inputs into a marketing prompt."""
    prompt = f"Write a {tone} {content_type} to {audience} promoting a new {product_name}."
    if keywords:
        prompt += f" Include keywords: {keywords}."
    prompt += " Highlight key features, benefits, and include a clear call-to-action. Keep it concise and engaging."
    return prompt

# -----------------------------
# Generate multiple variations
# -----------------------------
def generate_content(prompt, max_length=120, num_variations=3, temperature=0.9, top_k=50, top_p=0.95):
    """Generate `num_variations` outputs for the given prompt."""
    encoded = tokenizer(prompt, return_tensors="pt", truncation=True)
    input_ids = encoded.input_ids.to(device)

    # generate
    outputs = model.generate(
        input_ids,
        max_length=max_length,
        num_return_sequences=num_variations,
        do_sample=True,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        no_repeat_ngram_size=2,
        pad_token_id=tokenizer.pad_token_id
    )

    variations = []
    for out in outputs:
        text = tokenizer.decode(out, skip_special_tokens=True)
        # remove the original prompt if repeated
        if text.startswith(prompt):
            text = text[len(prompt):].strip()
        variations.append(text)
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
        content_type = st.selectbox("Content type", ["social post", "ad copy", "product description", "email subject"], index=0)
        audience = st.text_input("Target audience (optional)", value="customers")
        keywords = st.text_input("Keywords (comma separated)")
        tones = st.multiselect("Select tone(s)", default_tones, default=["Friendly"])
    with col2:
        st.write("\n")
        if st.button("Generate for single product"):
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
        if "Product" not in df_products.columns:
            st.error("CSV must contain a 'Product' column.")
        else:
            batch_keywords = st.text_input("Keywords (applied to all)", value="")
            batch_tone = st.selectbox("Tone for batch", ["Friendly", "Professional", "Humorous", "Persuasive"], index=0)
            batch_content_type = st.selectbox("Content type for batch", ["social post", "ad copy", "product description", "email subject"], index=0)
            batch_run = st.button("Generate batch content")
            if batch_run:
                generated = []
                with st.spinner("Generating batch content..."):
                    for _, row in df_products.iterrows():
                        p = str(row["Product"]).strip()
                        if not p:
                            continue
                        prompt = create_full_prompt(p, batch_tone, batch_keywords, batch_content_type)
                        variations = generate_content(
                            prompt,
                            max_length=max_length,
                            num_variations=num_variations,
                            temperature=temperature,
                            top_k=top_k,
                            top_p=top_p
                        )
                        for i, v in enumerate(variations, 1):
                            generated.append({
                                "Product": p,
                                "Tone": batch_tone,
                                "Variation": i,
                                "Content": v
                            })
                if generated:
                    df_generated = pd.DataFrame(generated)
                    st.success("Batch content generated!")
                    st.dataframe(df_generated)
                    csv = df_generated.to_csv(index=False)
                    st.download_button("Download batch results as CSV", csv, file_name="batch_marketing.csv", mime="text/csv")
