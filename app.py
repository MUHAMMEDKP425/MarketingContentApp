import streamlit as st
import pandas as pd
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch

# -----------------------------
# Load model (cached)
# -----------------------------
@st.cache_resource
def load_model(model_name="gpt2"):
    """Load tokenizer and model (CPU/GPU auto)."""
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    model = GPT2LMHeadModel.from_pretrained(model_name)
    model.resize_token_embeddings(len(tokenizer))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    return tokenizer, model, device

tokenizer, model, device = load_model()

# -----------------------------
# Prompt creation
# -----------------------------
def create_full_prompt(product_name, features="", tone="Friendly", content_type="social post"):
    examples = (
        "Example 1:\n"
        "Product: Electric Fan\n"
        "Tone: Friendly\n"
        "Features: quiet, energy-saving\n"
        "Post: \"Stay cool this summer with our energy-saving quiet electric fan. Order now!\"\n\n"
        "Example 2:\n"
        "Product: LED Lamp\n"
        "Tone: Professional\n"
        "Features: long-lasting, energy-efficient\n"
        "Post: \"Illuminate your space with our professional-grade LED lamp, energy-efficient and long-lasting. Buy today!\"\n\n"
    )

    base = f"Product: {product_name}\nTone: {tone}\n"
    if features:
        base += f"Features: {features}\n"
    base += f"Post:"

    full_prompt = examples + base
    return full_prompt

# -----------------------------
# Generate content
# -----------------------------
def generate_content(prompt, max_length=120, num_variations=3, temperature=0.8, top_k=50, top_p=0.9):
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
        pad_token_id=tokenizer.pad_token_id
    )

    variations = []
    for out in outputs:
        text = tokenizer.decode(out, skip_special_tokens=True)
        # remove everything up to the last "Post:" marker
        if "Post:" in text:
            text = text.split("Post:")[-1].strip()
        # keep first 2 sentences
        sentences = [s.strip() for s in text.replace("\n", " ").split(". ") if s.strip()]
        short = ". ".join(sentences[:2]).strip()
        if short and short[-1] not in ".!?":
            short += "."
        variations.append(short)
    return variations

# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(page_title="Professional Marketing Content Generator", layout="wide")
st.title("GPT-2 Marketing Content Generator")
st.caption("Generate relevant marketing posts for any product using features and tone. Works fully offline.")

# Sidebar settings
with st.sidebar:
    st.header("Generation Settings")
    max_length = st.slider("Max length (tokens)", 60, 300, 120)
    num_variations = st.slider("Variations per product", 1, 5, 3)
    temperature = st.slider("Creativity (temperature)", 0.1, 1.5, 0.8)
    top_k = st.slider("Top-K (tokens)", 10, 200, 50)
    top_p = st.slider("Top-p (nucleus)", 0.5, 1.0, 0.9)

# Single product generation
st.subheader("Generate Content for a Product")
col1, col2 = st.columns([3,1])
with col1:
    product_name = st.text_input("Product name (e.g., 'Air Cooler')")
    features = st.text_area("Key features (comma separated, optional)", placeholder="e.g., energy-efficient, quiet, remote control")
    tone = st.selectbox("Tone", ["Friendly", "Professional", "Humorous", "Persuasive"])
    content_type = st.selectbox("Content type", ["social post", "ad copy", "product description", "email subject"])
with col2:
    st.write("\n")
    if st.button("Generate Content"):
        if not product_name.strip():
            st.error("Please enter a product name.")
        else:
            prompt = create_full_prompt(product_name.strip(), features.strip(), tone, content_type)
            with st.spinner("Generating content..."):
                results = generate_content(prompt, max_length=max_length, num_variations=num_variations, temperature=temperature, top_k=top_k, top_p=top_p)
            df = pd.DataFrame({"Variation": list(range(1,len(results)+1)), "Content": results})
            st.success("Content generated!")
            st.dataframe(df)
            csv = df.to_csv(index=False)
            st.download_button("Download CSV", csv, file_name=f"{product_name.replace(' ','_')}_marketing.csv", mime="text/csv")
