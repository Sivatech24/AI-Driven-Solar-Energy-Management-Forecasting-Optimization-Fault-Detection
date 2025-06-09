import tkinter as tk
from tkinter import scrolledtext, END
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# Load model from local path or Hugging Face
model_path = "TinyLlama-1.1B-Chat-v1.0"  # Or local path
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float32)

# Set pad token if missing (important for attention mask)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Chat prompt format for TinyLLaMA
def format_prompt(user_message):
    return f"<|system|>\nYou are a helpful assistant.\n<|user|>\n{user_message}\n<|assistant|>\n"

# GUI Setup
root = tk.Tk()
root.title("TinyLLaMA Chat")

chat_display = scrolledtext.ScrolledText(root, wrap=tk.WORD, width=80, height=20)
chat_display.pack(padx=10, pady=10)
chat_display.insert(tk.END, "TinyLLaMA is ready. Ask me anything!\n\n")

user_input = tk.Entry(root, width=80)
user_input.pack(padx=10, pady=5)

def send_message():
    prompt = user_input.get().strip()
    if not prompt:
        return

    chat_display.insert(END, f"You: {prompt}\n")
    user_input.delete(0, END)

    full_prompt = format_prompt(prompt)
    inputs = tokenizer(full_prompt, return_tensors="pt", padding=True)
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]

    with torch.no_grad():
        output_ids = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=150,
            temperature=0.7,
            top_k=50,
            top_p=0.95,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    response = tokenizer.decode(output_ids[0][input_ids.shape[-1]:], skip_special_tokens=True)
    chat_display.insert(END, f"TinyLLaMA: {response.strip()}\n\n")
    chat_display.see(END)

send_button = tk.Button(root, text="Send", command=send_message)
send_button.pack(pady=5)

root.mainloop()
