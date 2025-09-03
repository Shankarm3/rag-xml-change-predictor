import tkinter as tk
from tkinter import ttk, filedialog, scrolledtext
from PIL import Image, ImageTk
import google.generativeai as genai
import os

class ImageAnalyzerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Image Analysis with Gemini")
        self.root.geometry("900x700")
        
        # Configure styles
        self.style = ttk.Style()
        self.style.configure('TButton', padding=6, relief="flat", background="#ccc")
        self.style.configure('TLabel', padding=6)
        
        # Configure grid
        self.root.grid_columnconfigure(0, weight=1)
        self.root.grid_rowconfigure(1, weight=1)
        
        # Header
        self.header = ttk.Label(
            self.root,
            text="Image Analysis with Google Gemini",
            font=('Helvetica', 16, 'bold')
        )
        self.header.grid(row=0, column=0, pady=10, sticky="n")
        
        # Main content frame
        self.main_frame = ttk.Frame(self.root, padding="10")
        self.main_frame.grid(row=1, column=0, sticky="nsew")
        self.main_frame.grid_columnconfigure(0, weight=1)
        self.main_frame.grid_rowconfigure(1, weight=1)
        
        # Upload button
        self.upload_btn = ttk.Button(
            self.main_frame,
            text="Upload Image",
            command=self.upload_image
        )
        self.upload_btn.grid(row=0, column=0, pady=10, sticky="ew")
        
        # Image display
        self.image_label = ttk.Label(self.main_frame)
        self.image_label.grid(row=1, column=0, pady=10)
        
        # Analysis result
        self.result_label = ttk.Label(
            self.main_frame,
            text="Analysis Result:",
            font=('Helvetica', 12, 'bold')
        )
        self.result_label.grid(row=2, column=0, pady=(20, 5), sticky="w")
        
        # Scrollable text area for results
        self.result_text = scrolledtext.ScrolledText(
            self.main_frame,
            wrap=tk.WORD,
            width=80,
            height=15,
            font=('Arial', 10)
        )
        self.result_text.grid(row=3, column=0, sticky="nsew", pady=5)
        
        # Status bar
        self.status_var = tk.StringVar()
        self.status_var.set("Ready")
        self.status_bar = ttk.Label(
            self.root,
            textvariable=self.status_var,
            relief=tk.SUNKEN,
            anchor=tk.W
        )
        self.status_bar.grid(row=2, column=0, sticky="ew")
        
        # Initialize Gemini
        try:
            genai.configure(api_key="AIzaSyCHDL4_qL_8fMG8aZM9co-1lP1ACYYXHqE")
            self.model = genai.GenerativeModel("gemini-1.5-flash")
            self.status_var.set("Gemini model loaded successfully")
        except Exception as e:
            self.status_var.set(f"Error loading Gemini: {str(e)}")
    
    def upload_image(self):
        file_path = filedialog.askopenfilename(
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.gif *.bmp")]
        )
        
        if not file_path:
            return
            
        try:
            # Display the image
            self.status_var.set("Processing image...")
            self.root.update()
            
            # Load and display image
            img = Image.open(file_path)
            img.thumbnail((600, 400))  # Resize for display
            photo = ImageTk.PhotoImage(img)
            self.image_label.config(image=photo)
            self.image_label.image = photo  # Keep a reference!
            
            # Analyze the image
            self.analyze_image(file_path)
            
        except Exception as e:
            self.status_var.set(f"Error: {str(e)}")
            self.result_text.delete(1.0, tk.END)
            self.result_text.insert(tk.END, f"Error processing image: {str(e)}")
    
    def analyze_image(self, image_path):
        try:
            self.status_var.set("Analyzing image with Gemini...")
            self.root.update()
            
            # Read image data
            with open(image_path, "rb") as f:
                img_data = f.read()
            
            # Generate content
            response = self.model.generate_content([
                {"mime_type": "image/jpeg" if image_path.lower().endswith(('.jpg', '.jpeg')) else "image/png",
                 "data": img_data},
                {"text": "Describe this image in detail."}
            ])
            
            # Display results
            self.result_text.delete(1.0, tk.END)
            self.result_text.insert(tk.END, response.text)
            self.status_var.set(f"Analysis complete - {os.path.basename(image_path)}")
            
        except Exception as e:
            self.status_var.set("Error during analysis")
            self.result_text.delete(1.0, tk.END)
            self.result_text.insert(tk.END, f"Error analyzing image: {str(e)}")

if __name__ == "__main__":
    root = tk.Tk()
    app = ImageAnalyzerApp(root)
    root.mainloop()