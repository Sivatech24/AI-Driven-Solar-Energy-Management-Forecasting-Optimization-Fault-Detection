import tkinter as tk
from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch

# Load model and tokenizer
model_name = "google/flan-t5-small"
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name)

# Chatbot logic
def generate_response():
    user_input = user_entry.get()
    if not user_input.strip():
        return

    # Prepare input for FLAN-T5
    input_text = "Answer the following question: " + user_input
    input_ids = tokenizer.encode(input_text, return_tensors="pt", truncation=True)

    # Generate response
    with torch.no_grad():
        output_ids = model.generate(input_ids, max_length=100)
    response = tokenizer.decode(output_ids[0], skip_special_tokens=True)

    # Display in chat window
    chat_log.config(state=tk.NORMAL)
    chat_log.insert(tk.END, "You: " + user_input + "\n")
    chat_log.insert(tk.END, "Bot: " + response + "\n\n")
    chat_log.config(state=tk.DISABLED)
    chat_log.yview(tk.END)

    user_entry.delete(0, tk.END)

# GUI setup
root = tk.Tk()
root.title("FLAN-T5 Chatbot")

chat_log = tk.Text(root, state=tk.DISABLED, wrap=tk.WORD, bg="white", fg="black", font=("Arial", 12))
chat_log.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)

user_entry = tk.Entry(root, font=("Arial", 12))
user_entry.pack(padx=10, pady=(0, 10), fill=tk.X)
user_entry.bind("<Return>", lambda event: generate_response())

send_button = tk.Button(root, text="Send", command=generate_response, font=("Arial", 12))
send_button.pack(padx=10, pady=(0, 10))

root.geometry("500x500")
root.mainloop()
