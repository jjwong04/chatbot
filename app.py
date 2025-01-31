from flask import Flask, request, jsonify, render_template
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

app = Flask(__name__)

# Predefined questions and answers
PREDEFINED_QUESTIONS = {
    "What are your hours of operation?": "We are open from 9:00 AM to 8:00 PM, Monday to Saturday! 🕒",
    "Where are you located?": "We are located at 123 Main Street, Springfield. 🗺️",
    "Do you offer delivery?": "Yes, we offer delivery within a 5-mile radius! 🚚",
    "What’s on the menu?": "Our menu includes pizza, pasta, salads, and desserts. 🍕🍝🥗🍰"
}

# Load LLaMA model and tokenizer (replace with your model path)
MODEL_NAME = "gpt2"  # Example LLaMA model on Hugging Face
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, low_cpu_mem_usage=True, torch_dtype="auto")

def get_ai_response(user_input):
    """Get a response from the LLaMA model for non-predefined questions."""
    inputs = tokenizer(user_input, return_tensors="pt")
    outputs = model.generate(**inputs, max_length=100)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/chat", methods=["POST"])
def chat():
    user_input = request.json.get("message")
    if user_input in PREDEFINED_QUESTIONS:
        response = PREDEFINED_QUESTIONS[user_input]
    else:
        response = get_ai_response(user_input)
    return jsonify({"response": response})

if __name__ == "__main__":
    app.run(debug=True)