import tkinter as tk
from tkinter import scrolledtext
import google.generativeai as genai
import time
import re

API_KEY = "AIzaSyCHDL4_qL_8fMG8aZM9co-1lP1ACYYXHqE"
genai.configure(api_key=API_KEY)

model = genai.GenerativeModel("gemini-1.5-flash")

def clean_markdown(text):
    """Convert Markdown to plain text with bullet points & line breaks."""
    text = re.sub(r"(?m)^[\*\-]\s+", "• ", text)
    text = re.sub(r"\*\*(.*?)\*\*", r"\1", text)
    text = re.sub(r"\*(.*?)\*", r"\1", text)
    return text

def send_message(event=None):
    user_input = entry.get().strip()
    if not user_input:
        return

    add_message("You", user_input, "user")
    entry.delete(0, tk.END)

    chat_box.config(state=tk.NORMAL)
    chat_box.insert(tk.END, "Gemini:\n", "bot_name")
    chat_box.insert(tk.END, "", "bot")
    chat_box.config(state=tk.DISABLED)
    chat_box.yview(tk.END)

    stream_response(user_input)

def stream_response(user_input):
    chat_box.config(state=tk.NORMAL)
    try:
        response = model.generate_content(user_input, stream=True)

        for chunk in response:
            if chunk.candidates:
                text_chunk = chunk.candidates[0].content.parts[0].text
                text_chunk = clean_markdown(text_chunk)

                # Split by line, not just spaces
                for line in text_chunk.split("\n"):
                    words = line.split(" ")
                    for word in words:
                        chat_box.insert(tk.END, word + " ", "bot")
                        chat_box.yview(tk.END)
                        root.update_idletasks()
                        time.sleep(0.02)
                    chat_box.insert(tk.END, "\n", "bot")  # Add line break

    except Exception as e:
        chat_box.insert(tk.END, f"Error: {e}", "bot")

    chat_box.insert(tk.END, "\n", "bot")
    chat_box.config(state=tk.DISABLED)

def add_message(sender, message, tag):
    chat_box.config(state=tk.NORMAL)
    chat_box.insert(tk.END, f"{sender}:\n", f"{tag}_name")
    chat_box.insert(tk.END, f"{message}\n\n", tag)
    chat_box.config(state=tk.DISABLED)
    chat_box.yview(tk.END)

root = tk.Tk()
root.title("Gemini AI Chat (Streaming)")
root.geometry("700x550")
root.configure(bg="#1E1E2E")

root.grid_rowconfigure(0, weight=1)
root.grid_columnconfigure(0, weight=1)

chat_box = scrolledtext.ScrolledText(root, wrap=tk.WORD, font=("Segoe UI", 12),
                                     state=tk.DISABLED, bg="#252538", fg="white",
                                     insertbackground="white", bd=0, relief=tk.FLAT)
chat_box.grid(row=0, column=0, columnspan=2, padx=15, pady=15, sticky="nsew")

chat_box.tag_config("user_name", foreground="#00BFFF", font=("Segoe UI", 10, "bold"))
chat_box.tag_config("bot_name", foreground="#32CD32", font=("Segoe UI", 10, "bold"))
chat_box.tag_config("user", foreground="white", font=("Segoe UI", 12), lmargin1=10, lmargin2=10)
chat_box.tag_config("bot", foreground="#E5E5E5", font=("Segoe UI", 12), lmargin1=10, lmargin2=10)

entry = tk.Entry(root, font=("Segoe UI", 14), bg="#2C2C3E", fg="white",
                 insertbackground="white", relief=tk.FLAT, bd=2)
entry.grid(row=1, column=0, padx=(15, 5), pady=(0, 15), sticky="ew", ipady=8)
entry.focus()

send_btn = tk.Button(root, text="Send", command=send_message, font=("Segoe UI", 12, "bold"),
                     bg="#00BFFF", fg="white", relief=tk.FLAT, padx=20, pady=8, cursor="hand2")
send_btn.grid(row=1, column=1, padx=(5, 15), pady=(0, 15), sticky="ew")

root.bind("<Return>", send_message)

root.mainloop()
