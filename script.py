import streamlit as st
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Set model name (replace this with the actual model you're using)
MODEL_NAME = "facebook/opt-1.3b"  # Change if using another model

# Function to load model safely
@st.cache_resource
def load_model():
    try:
        st.write(f"Loading model: {MODEL_NAME}")
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float16, device_map="auto")
        st.write("✅ Model loaded successfully!")
        return model, tokenizer
    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
        return None, None

# Load the model
model, tokenizer = load_model()

# Streamlit UI
st.title("OmniMind - AI Personality & Future Predictor")
user_input = st.text_input("Ask me anything:")

# Generate response
if st.button("Generate Response") and model and tokenizer:
    with st.spinner("Thinking..."):
        try:
            inputs = tokenizer(user_input, return_tensors="pt").to("cuda")
            output = model.generate(**inputs, max_length=50)
            response = tokenizer.decode(output[0], skip_special_tokens=True)
            st.success("OmniMind's Response:")
            st.write(response)
        except Exception as e:
            st.error(f"❌ Error generating response: {e}")
else:
    st.warning("Please enter a question and make sure the model is loaded.")
