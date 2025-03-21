import streamlit as st
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load AI Model
MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.2"  # Change this if needed
device = "cuda" if torch.cuda.is_available() else "cpu"

@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME).to(device)
    return model, tokenizer

model, tokenizer = load_model()

# OmniMind: Generate Conversational Response
def generate_response(prompt):
    """Generates a normal AI response based on user input."""
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    output = model.generate(
        input_ids, 
        max_length=150, 
        temperature=0.7, 
        do_sample=True, 
        top_k=50, 
        top_p=0.9
    )
    response = tokenizer.decode(output[0], skip_special_tokens=True)
    
    # Cleaning up output
    response = response.replace("\n", " ").strip()
    
    # Prevent broken responses
    if len(response) < 5 or "." not in response:
        response = "I'm not sure, could you clarify?"
    
    return response

# Multiverse-Based Future Prediction
def generate_future():
    """Generates possible futures based on probability (multiverse concept)."""
    futures = [
        "You become extremely successful in your field.",
        "You face challenges but achieve moderate success.",
        "You discover an unexpected opportunity leading to great success.",
        "You take a different path that leads to an unconventional but fulfilling career.",
        "You struggle to find your path but eventually overcome difficulties."
    ]
    
    probabilities = torch.softmax(torch.randn(len(futures)), dim=0)  # Generates realistic probabilities
    
    # Format output with probabilities
    future_predictions = [
        f"{future} (Probability: {prob:.2%})"
        for future, prob in zip(futures, probabilities)
    ]
    
    return future_predictions

# Streamlit UI
st.title("🤖 OmniMind - AI Chat & Future Scopes")

tab1, tab2 = st.tabs(["Chat with OmniMind", "Multiverse Future Scopes"])

# Chatbot Interface
with tab1:
    user_input = st.text_input("Ask OmniMind anything:")
    if user_input:
        response = generate_response(user_input)
        st.write(response)

# Future Scopes
with tab2:
    if st.button("Generate Possible Futures"):
        possible_futures = generate_future()
        st.write("\n".join(possible_futures))
