import streamlit as st
import pandas as pd
from transformers import pipeline, set_seed
import io

# ----------------------------
# CONFIG
# ----------------------------
st.set_page_config(
    page_title="Marketing Content Generator",
    page_icon="💡",
    layout="centered"
)

st.title("💡 Marketing Content Generator")
st.write("Generate professional and engaging marketing content for your products using AI!")

# ----------------------------
# LOAD MODEL
# ----------------------------
@st.cache_resource
def load_model():
    generator = pipeline("text-generation", model="gpt2")
    set_seed(42)
    return generator

generator = load_model()

# ----------------------------
# HELPER FUNCTIONS
# ----------------------------
def create_prompt(product, features, tone, content_type, audience, keywords):
    prompt = f"""
You are a professional marketing copywriter.

Write a {tone.lower()} {content_type.lower()} for a product called "{product}".
Highlight these key features: {features if features else "great quality and value"}.
Target audience: {audience if audience else "general customers"}.
Focus on these keywords: {keywords if keywords else "premium, value, new launch"}.

Example:
Product: EcoCool Fan
Tone: Friendly
Content Type: Social Media Post
Output: "Stay cool effortlessly! 🌬️ Meet the EcoCool Fan – energy-efficient, whisper-quiet, and designed for comfort. Perfect for summer days. #CoolComfort #EcoCool"

Now generate one relevant and appealing {content_type.lower()}.
"""
    return prompt.strip()

def generate_content(prompt):
    output = generator(
        prompt,
        max_length=150,
        num_return_sequences=1,
        temperature=0.6,
        top_k=50,
        no_repeat_ngram_size=3
    )
    text = output[0]['generated_text']
    # Trim after first 2 sentences to keep it concise
    text = text.split(". ")
    return ". ".join(text[:2]).strip()

def safe_read_tabular(file):
    """Reads CSV or Excel with fallback encodings"""
    try:
        return pd.read_csv(file, encoding='utf-8')
    except Exception:
        file.seek(0)
        try:
            return pd.read_csv(file, encoding='latin1')
        except Exception:
            file.seek(0)
            return pd.read_excel(file)

# ----------------------------
# TABS
# ----------------------------
tab1, tab2 = st.tabs(["🧍 Single Product", "📦 Batch Upload"])

# ----------------------------
# TAB 1 - SINGLE PRODUCT
# ----------------------------
with tab1:
    st.subheader("Generate Content for a Single Product")

    product = st.text_input("Product Name")
    features = st.text_area("Key Features (comma-separated)")
    tone = st.selectbox("Tone", ["Friendly", "Professional", "Creative", "Persuasive"])
    content_type = st.selectbox("Content Type", ["Social Media Post", "Ad Copy", "Email", "Product Description"])
    audience = st.text_input("Target Audience (optional)", "Customers")
    keywords = st.text_input("Keywords (comma-separated, optional)")

    if st.button("🚀 Generate"):
        if product:
            with st.spinner("Generating content... please wait."):
                prompt = create_prompt(product, features, tone, content_type, audience, keywords)
                result = generate_content(prompt)
                st.success("✅ Generated Content:")
                st.write(result)
        else:
            st.warning("Please enter a product name before generating content.")

# ----------------------------
# TAB 2 - BATCH UPLOAD
# ----------------------------
with tab2:
    st.subheader("Generate Content for Multiple Products")
    uploaded_file = st.file_uploader("Upload CSV or Excel file", type=["csv", "xlsx"])

    if uploaded_file:
        try:
            data = safe_read_tabular(uploaded_file)
            st.write("📋 Uploaded Data Preview:")
            st.dataframe(data.head())

            if 'product' not in data.columns:
                st.error("Your file must contain a column named 'product'.")
            else:
                if st.button("⚡ Generate Batch Content"):
                    with st.spinner("Generating batch content... please wait."):
                        results = []
                        for _, row in data.iterrows():
                            product = row['product']
                            features = row.get('features', '')
                            tone = row.get('tone', 'Friendly')
                            content_type = row.get('content_type', 'Social Media Post')
                            audience = row.get('audience', 'Customers')
                            keywords = row.get('keywords', '')

                            prompt = create_prompt(product, features, tone, content_type, audience, keywords)
                            content = generate_content(prompt)
                            results.append({
                                "product": product,
                                "generated_content": content
                            })

                        output_df = pd.DataFrame(results)
                        st.success("✅ Content Generated Successfully!")
                        st.dataframe(output_df)

                        csv = output_df.to_csv(index=False).encode('utf-8')
                        st.download_button("📥 Download Results as CSV", data=csv, file_name="generated_marketing_content.csv")

        except Exception as e:
            st.error(f"Error reading file: {e}")
