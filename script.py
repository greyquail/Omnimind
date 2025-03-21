import streamlit as st
import torch
import random
from transformers import AutoTokenizer, AutoModelForCausalLM

# --------------------------------------------------------------
# Load AI Model (GPT-Neo 125M)
# --------------------------------------------------------------
@st.cache_resource(show_spinner=True)
def load_model():
    model_name = "EleutherAI/gpt-neo-125M"  # Small model for efficiency
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float32,  # Use float32 on CPU for stability
        low_cpu_mem_usage=True
    )
    return tokenizer, model

# Load Model
tokenizer, model = load_model()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# --------------------------------------------------------------
# Generate Multiple Futures (Multiverse Concept)
# --------------------------------------------------------------
def generate_multiverse_possibilities(prompt):
    """Generate different possible outcomes with probabilities."""
    futures = [
        ("A breakthrough success in your field", 0.35),
        ("A slow and steady rise to success", 0.25),
        ("A challenging journey with obstacles", 0.20),
        ("Unexpected setbacks leading to a career shift", 0.10),
        ("An unforeseen opportunity changes your path", 0.10),
    ]
    
    random.shuffle(futures)  # Randomize order to make it dynamic
    return futures

# --------------------------------------------------------------
# AI Chatbot Function
# --------------------------------------------------------------
def generate_response(prompt):
    """Generate AI-based conversational responses."""
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    output = model.generate(input_ids, max_length=150, temperature=0.7, do_sample=True)
    response = tokenizer.decode(output[0], skip_special_tokens=True)
    return response

# --------------------------------------------------------------
# Streamlit UI
# --------------------------------------------------------------
st.title("🧠 OmniMind: Your AI Companion")

st.write("Ask anything, and I’ll respond like a conversational AI. You can also ask about possible future outcomes!")

user_input = st.text_input("You:", "")

if st.button("Ask"):
    if user_input.lower().startswith("what are my possible futures"):
        possibilities = generate_multiverse_possibilities(user_input)
        st.subheader("🌟 Possible Futures:")
        for future, probability in possibilities:
            st.write(f"- **{future}** *(Probability: {probability * 100:.0f}%)*")
    else:
        response = generate_response(user_input)
        st.subheader("🤖 OmniMind:")
        st.write(response)
