import streamlit as st
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load AI Model
@st.cache_resource
def load_model():
    try:
        model_name = "mistralai/Mistral-7B-Instruct"  # Example model (adjust as needed)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        return model, tokenizer
    except Exception as e:
        st.error(f"Failed to load AI model: {e}")
        return None, None

# Initialize model
model, tokenizer = load_model()

st.title("🔮 OmniMind - Personality & Future Predictor")

# Ensure model loaded successfully
if model is None:
    st.error("Error: AI model not loaded.")
else:
    st.success("AI Model Loaded Successfully!")

# User Input
user_input = st.text_input("Enter your question:")
if st.button("Generate Response"):
    if model is not None and tokenizer is not None:
        try:
            inputs = tokenizer(user_input, return_tensors="pt").to("cuda" if torch.cuda.is_available() else "cpu")
            output = model.generate(**inputs, max_length=100)
            response = tokenizer.decode(output[0], skip_special_tokens=True)
            st.write(f"OmniMind's Response: {response}")
        except Exception as e:
            st.error(f"Error generating response: {e}")
    else:
        st.error("AI model is not loaded. Please check logs.")
