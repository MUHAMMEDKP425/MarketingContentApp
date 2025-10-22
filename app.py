import streamlit as st
import pandas as pd
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# -----------------------------
# Load GPT-2 model
# -----------------------------
@st.cache_resource
def load_model():
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    model = GPT2LMHeadModel.from_pretrained("gpt2")
    return tokenizer, model

tokenizer, model = load_model()

# -----------------------------
# Function to generate multiple variations
# -----------------------------
def generate_content(prompt, max_length=100, num_variations=3):
    inputs = tokenizer.encode(prompt, return_tensors="pt")
    outputs = model.generate(
        inputs,
        max_length=max_length,
        num_return_sequences=num_variations,
        no_repeat_ngram_size=2,
        do_sample=True,
        top_k=50,
        top_p=0.95
    )
    return [tokenizer.decode(out, skip_special_tokens=True) for out in outputs]

# -----------------------------
# Function to create full prompt
# -----------------------------
def create_full_prompt(product_name, tone="Friendly", keywords=""):
    prompt = f"Write a {tone} social media post for a new {product_name}."
    if keywords:
        prompt += f" Include keywords: {keywords}."
    prompt += " Highlight key features and add a call-to-action."
    return prompt

# -----------------------------
# Streamlit UI
# -----------------------------
st.title("Professional Marketing Content Generator")

# Tabs for Single Product vs Batch Upload
tab1, tab2 = st.tabs(["Single Product", "Batch Upload"])

# -----------------------------
# Tab 1: Single Product
# -----------------------------
with tab1:
    st.subheader("Generate content for a single product")
    user_input = st.text_input("Enter product name:")
    keywords = st.text_input("Enter keywords (comma separated)")
    tone = st.selectbox("Select Tone", ["Friendly", "Professional", "Humorous", "Persuasive"])
    max_length = st.slider("Max content length", 50, 300, 100)
    num_variations = st.slider("Number of variations", 1, 5, 3)

    if st.button("Generate Content"):
        if user_input:
            full_prompt = create_full_prompt(user_input, tone, keywords)
            with st.spinner("Generating content..."):
                results = generate_content(full_prompt, max_length, num_variations)
            st.success("Content Generated!")
            for i, r in enumerate(results, 1):
                st.write(f"**Variation {i}:** {r}")
        else:
            st.error("Please enter a product name!")

# -----------------------------
# Tab 2: Batch Upload (CSV)
# -----------------------------
with tab2:
    st.subheader("Generate content for multiple products from CSV")
    uploaded_file = st.file_uploader("Upload CSV with product names (column: 'Product')", type="csv")
    
    if uploaded_file:
        data = pd.read_csv(uploaded_file)
        if "Product" not in data.columns:
            st.error("CSV must have a column named 'Product'")
        else:
            keywords_batch = st.text_input("Enter keywords (applied to all products)")
            tone_batch = st.selectbox("Select Tone for all products", ["Friendly", "Professional", "Humorous", "Persuasive"])
            max_length_batch = st.slider("Max content length", 50, 300, 100)
            num_variations_batch = st.slider("Number of variations per product", 1, 5, 3)

            if st.button("Generate Batch Content"):
                generated_data = []
                with st.spinner("Generating content for all products..."):
                    for _, row in data.iterrows():
                        prompt = create_full_prompt(row["Product"], tone_batch, keywords_batch)
                        variations = generate_content(prompt, max_length_batch, num_variations_batch)
                        for v in variations:
                            generated_data.append({"Product": row["Product"], "Tone": tone_batch, "Content": v})
                df = pd.DataFrame(generated_data)
                st.dataframe(df)
                st.download_button(
                    label="Download Generated Content",
                    data=df.to_csv(index=False),
                    file_name="marketing_content.csv",
                    mime="text/csv"
                )
                st.success("All content generated and ready to download!")
