import streamlit as st
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Choose the AI Model (Try different ones if needed)
MODEL_NAME = "mistralai/Mistral-7B-Instruct"  # Alternative: "microsoft/DialoGPT-medium"

# Load Model and Tokenizer
@st.cache_resource()
def load_model():
    try:
        print("Loading model...")
        model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float16, device_map="auto")
        print(f"Model loaded successfully: {MODEL_NAME}")

        print("Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        print("Tokenizer loaded successfully!")
        
        return model, tokenizer
    except Exception as e:
        print(f"Error loading model: {e}")
        st.error("Failed to load AI model. Check the logs.")
        return None, None

model, tokenizer = load_model()

# Generate Response Function
def generate_response(prompt):
    if model is None or tokenizer is None:
        return "Error: AI model not loaded."

    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    
    output = model.generate(
        **inputs,
        max_length=150,   # Increased response length
        do_sample=True,   # Enables natural variation
        temperature=0.7,  # Controls randomness
        top_p=0.9         # Filters low probability words
    )
    
    response = tokenizer.decode(output[0], skip_special_tokens=True)
    return response.strip()

# Streamlit UI
st.title("🔮 OmniMind - Personality & Future Predictor")
st.write("Ask me anything!")

user_input = st.text_input("Enter your question:")
if st.button("Generate Response"):
    if user_input.strip():
        response = generate_response(user_input)
        st.success("OmniMind's Response:")
        st.write(response)
    else:
        st.warning("Please enter a valid question.")
