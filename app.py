import streamlit as st
import pandas as pd
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch
from sentence_transformers import SentenceTransformer
import io

# -----------------------------
# Load models
# -----------------------------
@st.cache_resource
def load_gpt2(model_name="gpt2"):
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    model = GPT2LMHeadModel.from_pretrained(model_name)
    model.resize_token_embeddings(len(tokenizer))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    return tokenizer, model, device

tokenizer, model, device = load_gpt2()

@st.cache_resource
def load_embedding_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

embedding_model = load_embedding_model()

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
        return pd.read_excel(io.BytesIO(raw))
    encodings_to_try = ["utf-8", "utf-8-sig", "latin1", "cp1252", "iso-8859-1"]
    for enc in encodings_to_try:
        try:
            s = raw.decode(enc)
            return pd.read_csv(io.StringIO(s))
        except Exception:
            continue
    return pd.read_csv(io.StringIO(raw.decode("utf-8", errors="replace")), engine="python")

# -----------------------------
# GPT-2 Content Generation
# -----------------------------
def create_full_prompt(product_name, tone="Friendly", keywords="", content_type="social post", audience="customers", features=""):
    base = f"Write a {tone} {content_type} to {audience} promoting a new {product_name}."
    if features:
        base += f" Key features: {features}."
    if keywords:
        base += f" Include keywords: {keywords}."
    base += " Keep it concise (1-2 short sentences), highlight benefits, and add a clear call-to-action."
    full_prompt = base + "\nPost:"
    return full_prompt

def generate_content(prompt, max_length=100, num_variations=3, temperature=0.65, top_k=40, top_p=0.9):
    encoded = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=model.config.n_positions - 1)
    input_ids = encoded.input_ids.to(device)
    max_gen_length = min(max_length, model.config.n_positions - input_ids.shape[1] - 1)
    if max_gen_length <= 0:
        raise ValueError("Input prompt too long.")
    outputs = model.generate(
        input_ids,
        max_length=input_ids.shape[1] + max_gen_length,
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
# Embedding Search Functions
# -----------------------------
def compute_embeddings(df, column='Content'):
    texts = df[column].astype(str).tolist()
    embeddings = embedding_model.encode(texts, convert_to_tensor=True)
    return embeddings

def search_relevant(query, df, embeddings, top_k=5):
    query_emb = embedding_model.encode([query], convert_to_tensor=True)
    cos_scores = torch.nn.functional.cosine_similarity(query_emb, embeddings)
    top_results = torch.topk(cos_scores, k=min(top_k, len(df)))
    results = df.iloc[top_results.indices.cpu().numpy()]
    return results

def generate_from_dataset(query, retrieved_texts, tone='Friendly'):
    context = "\n".join(retrieved_texts)
    prompt = f"I have the following posts about similar products:\n{context}\n\nNow write a {tone} social post about {query}, highlighting key features and benefits. Keep it concise."
    return generate_content(prompt, max_length=120, num_variations=3)

# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(page_title="Marketing Content Generator", layout="wide")
st.title("Professional Marketing Content Generator")
st.caption("Generate content from your dataset or create new posts.")

# Sidebar
with st.sidebar:
    max_length = st.slider("Max length (tokens)", 60, 300, 120)
    num_variations = st.slider("Variations per prompt", 1, 5, 3)
    temperature = st.slider("Creativity (temperature)", 0.1, 1.5, 0.9)
    top_k = st.slider("Top-K (tokens)", 10, 200, 50)
    top_p = st.slider("Top-p (nucleus)", 0.5, 1.0, 0.95)

# Tabs
tab1, tab2, tab3 = st.tabs(["Single Product", "Batch Upload", "Search Dataset"])

# Single Product Tab
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

# Batch Upload Tab
with tab2:
    st.subheader("Batch generation from CSV")
    uploaded_file = st.file_uploader("Upload CSV or Excel (must have 'Product' column)", type=["csv","xls","xlsx"])
    if uploaded_file:
        df_products = safe_read_tabular(uploaded_file)
        if 'Product' in [c.lower() for c in df_products.columns]:
            if st.button("Generate batch content"):
                generated = []
                for _, row in df_products.iterrows():
                    p = str(row['Product']).strip()
                    if not p:
                        continue
                    features_val = str(row.get('features','') or '')
                    tone_val = str(row.get('tone','') or 'Friendly')
                    content_type_val = str(row.get('content_type','') or 'social post')
                    audience_val = str(row.get('audience','') or 'customers')
                    keywords_val = str(row.get('keywords','') or '')
                    prompt = create_full_prompt(p, tone_val, keywords_val, content_type_val, audience_val, features_val)
                    variations = generate_content(prompt, max_length=max_length, num_variations=num_variations, temperature=temperature, top_k=top_k, top_p=top_p)
                    for i, v in enumerate(variations,1):
                        generated.append({"Product": p, "Tone": tone_val, "Variation": i, "Content": v})
                df_out = pd.DataFrame(generated)
                st.dataframe(df_out)
                st.download_button("Download batch CSV", df_out.to_csv(index=False), file_name="batch_marketing_content.csv", mime="text/csv")
        else:
            st.error("CSV must contain a 'Product' column.")

# Search Dataset Tab
with tab3:
    st.subheader("Search your uploaded dataset")
    dataset_file = st.file_uploader("Upload CSV/Excel with a 'Content' column", type=["csv","xls","xlsx"], key="dataset")
    if dataset_file:
        df_dataset = safe_read_tabular(dataset_file)
        if 'Content' not in df_dataset.columns:
            st.error("Dataset must have a 'Content' column.")
        else:
            embeddings = compute_embeddings(df_dataset, column='Content')
            query = st.text_input("Enter a search query")
            generate_new = st.checkbox("Generate new post based on retrieved content", value=True)
            if st.button("Search") and query:
                results = search_relevant(query, df_dataset, embeddings, top_k=5)
                st.subheader("Top Relevant Content")
                st.dataframe(results)
                if generate_new:
                    new_posts = generate_from_dataset(query, results['Content'].tolist(), tone='Friendly')
                    st.subheader("Generated Posts")
                    for i, post in enumerate(new_posts,1):
                        st.markdown(f"**Post {i}:** {post}")

st.markdown("---")
st.markdown("Built with GPT-2 (transformers) + SentenceTransformers for search.")
