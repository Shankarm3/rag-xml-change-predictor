import tkinter as tk
from tkinter import ttk, filedialog, scrolledtext, font, messagebox
import json
import asyncio
import threading
import os
import sys
from pathlib import Path

class StyledButton(ttk.Button):
    """Custom styled button with consistent padding and colors"""
    def __init__(self, master=None, **kwargs):
        super().__init__(master, **kwargs)
        self.configure(
            style='Accent.TButton',
            padding=10
        )

class XMLAnalyzerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("XML Change Predictor")
        self.root.geometry("1000x700")
        self.root.minsize(800, 600)
        
        # Configure styles
        self.style = ttk.Style()
        self.style.theme_use('clam')  # Use a modern theme
        
        # Custom colors
        self.bg_color = '#f0f2f5'
        self.primary_color = '#4a6fa5'
        self.secondary_color = '#6c757d'
        self.success_color = '#28a745'
        self.danger_color = '#dc3545'
        
        # Configure styles
        self.style.configure('.', background=self.bg_color)
        self.style.configure('TFrame', background=self.bg_color)
        self.style.configure('TLabel', background=self.bg_color, font=('Segoe UI', 12))
        self.style.configure('TButton', 
                           font=('Segoe UI', 10, 'bold'),
                           padding=8,
                           borderwidth=0)
        self.style.configure('Header.TLabel', 
                           font=('Segoe UI', 16, 'bold'),
                           foreground=self.primary_color)
        self.style.configure('Accent.TButton',
                           background=self.primary_color,
                           foreground='white',
                           borderwidth=0)
        self.style.map('Accent.TButton',
                      background=[('active', '#3a5a8f')])
        self.style.configure('TEntry', 
                           fieldbackground='white',
                           borderwidth=1,
                           relief='solid')
        
        # Improved StatusBar style (define before packing widgets!)
        self.status_var = tk.StringVar()
        self.status_var.set("Ready")
        self.style.configure('StatusBar.TLabel',
            background="#e4eef9",
            foreground="#196fa2",
            font=('Segoe UI', 12, 'bold'),
            padding=(13, 10, 8, 10),
            relief='ridge',
            borderwidth=2
        )
        self.status_bar = ttk.Label(
            self.root,  # Attach to root
            textvariable=self.status_var,
            anchor=tk.W,
            style='StatusBar.TLabel'
        )
        self.status_bar.pack(fill=tk.X, side=tk.BOTTOM, pady=(0, 0))

        # Main container
        self.main_frame = ttk.Frame(root, padding=(20, 15))
        self.main_frame.pack(fill=tk.BOTH, expand=True)
        
        # HERO CARD HEADER
        self.style.configure('HeroCard.TFrame', background='#fafdff', relief='ridge', borderwidth=3, bordercolor='#c5e6ff')
        self.style.configure('HeroTitle.TLabel', font=('Segoe UI', 26, 'bold'), foreground='#0185fc', background='#fafdff')

        hero_frame = ttk.Frame(self.main_frame, style='HeroCard.TFrame', padding=(35,25,35,25))
        hero_frame.pack(fill=tk.X, pady=(0, 27))
        ttk.Label(
            hero_frame,
            text="🧩 XML Change Predictor",
            style='HeroTitle.TLabel',
            anchor='center',
            justify='center',
            padding=(8,2,8,0)
        ).pack(anchor='center')
        ttk.Label(
            hero_frame,
            text="AI-powered suggestions for smarter XML evolution.",
            font=('Segoe UI', 14, 'italic'),
            foreground='#4ba2e8',
            background='#fafdff',
            anchor='center',
            justify='center',
            padding=(8,3,8,13)
        ).pack(anchor='center')
        ttk.Label(
            hero_frame,
            text="Upload an XML file to discover likely changes, frequent patterns, and improvement ideas — all powered by data-driven AI.",
            font=('Segoe UI', 11),
            foreground='#374e60',
            background='#fafdff',
            anchor='center',
            justify='center',
            wraplength=640,
            padding=(10,0,10,5)
        ).pack(anchor='center')
        # End HERO CARD HEADER

        # JOURNAL SELECTION
        journal_frame = ttk.LabelFrame(
            self.main_frame,
            text=" Select Journal (Data Source) ",
            padding=15,
            style='Card.TLabelframe'
        )
        journal_frame.pack(fill=tk.X, pady=(0, 15))

        self.available_journals = self.get_journal_folders()
        self.journal_var = tk.StringVar()
        self.journal_combo = ttk.Combobox(
            journal_frame,
            textvariable=self.journal_var,
            values=self.available_journals,
            state="readonly" if self.available_journals else "disabled",
            font=('Segoe UI', 11),
            width=45
        )
        self.journal_combo.pack(side=tk.LEFT, padx=(0, 10), fill=tk.X, expand=True)
        self.journal_combo.bind("<<ComboboxSelected>>", lambda e: self.on_journal_selected())
        if self.available_journals:
            self.status_var.set("Please select a journal before analysis.")
        else:
            self.status_var.set("No journals found under data/. Please add a journal folder.")

        # End JOURNAL SELECTION
        
        # File selection
        self.file_frame = ttk.LabelFrame(
            self.main_frame,
            text=" File Selection ",
            padding=15,
            style='Card.TLabelframe'
        )
        self.file_frame.pack(fill=tk.X, pady=(0, 15))
        
        # File entry with browse button
        input_frame = ttk.Frame(self.file_frame)
        input_frame.pack(fill=tk.X, pady=5)
        
        self.file_path = tk.StringVar()
        self.file_entry = ttk.Entry(
            input_frame,
            textvariable=self.file_path,
            font=('Segoe UI', 10)
        )
        self.file_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 10))
        
        self.browse_btn = StyledButton(
            input_frame,
            text="Browse...",
            command=self.browse_file,
            state=tk.DISABLED if not self.available_journals else tk.NORMAL
        )
        self.browse_btn.pack(side=tk.RIGHT)
        
        # Control buttons
        self.control_frame = ttk.Frame(self.main_frame)
        self.control_frame.pack(fill=tk.X, pady=(0, 15))
        
        btn_frame = ttk.Frame(self.control_frame)
        btn_frame.pack(side=tk.LEFT)
        
        self.analyze_btn = StyledButton(
            btn_frame,
            text="Start Analysis",
            command=self.start_analysis,
            state=tk.DISABLED if not self.available_journals else tk.NORMAL
        )
        self.analyze_btn.pack(side=tk.LEFT, padx=(0, 10))
        
        # Clear Button with improved custom style
        self.style.configure('Clear.TButton',
            background=self.danger_color,
            foreground='white',
            font=('Segoe UI', 10, 'bold'),
            padding=8,
            borderwidth=0,
            relief='flat',
        )
        self.style.map('Clear.TButton',
            background=[('active', '#ba2630'), ('pressed', '#ec7676')],
            foreground=[('active', 'white'), ('pressed', 'white')]
        )
        self.clear_btn = ttk.Button(
            btn_frame,
            text="Clear Results",
            command=self.clear_output,
            style='Clear.TButton'
        )
        self.clear_btn.pack(side=tk.LEFT)
        
        # Progress bar
        self.progress = ttk.Progressbar(
            self.control_frame,
            orient=tk.HORIZONTAL,
            length=600,
            mode='indeterminate',
            style='TProgressbar'
        )
        self.progress.pack(side=tk.RIGHT)
        
        # Output console with tabs
        self.notebook = ttk.Notebook(self.main_frame)
        self.notebook.pack(fill=tk.BOTH, expand=True, pady=(0, 10))
        
        # Console tab
        self.console_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.console_frame, text="Console Output")
        
        # JSON Results tab
        self.json_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.json_frame, text="JSON Results")
        
        # Console output
        self.output_text = scrolledtext.ScrolledText(
            self.console_frame,
            wrap=tk.WORD,
            font=('Consolas', 10),
            bg='white',
            fg='#333',
            insertbackground='#333',
            selectbackground='#e6f2ff',
            selectforeground='#333',
            padx=12,
            pady=12,
            bd=0,
            relief='flat',
            state=tk.DISABLED
        )
        self.output_text.pack(fill=tk.BOTH, expand=True)
        
        self.json_text = scrolledtext.ScrolledText(
            self.json_frame,
            wrap=tk.WORD,
            font=('Consolas', 9),
            bg='#f8f9fa',
            fg='#2c3e50',
            insertbackground='#2c3e50',
            selectbackground='#d6e4ff',
            selectforeground='#2c3e50',
            padx=12,
            pady=12,
            bd=0,
            relief='flat',
            state=tk.DISABLED
        )
        self.json_text.pack(fill=tk.BOTH, expand=True)
        
        self.json_text.tag_configure("key", foreground="#2c3e50", font=('Consolas', 9, 'bold'))
        self.json_text.tag_configure("string", foreground="#27ae60")
        self.json_text.tag_configure("number", foreground="#e74c3c")
        self.json_text.tag_configure("boolean", foreground="#8e44ad")
        self.json_text.tag_configure("null", foreground="#7f8c8d")
        self.json_text.tag_configure("bracket", foreground="#2c3e50")
        
        # (moved above)
        
        sys.stdout = TextRedirector(self.output_text, "stdout")
        
        self.main_frame.columnconfigure(0, weight=1)
        self.main_frame.rowconfigure(1, weight=1)
        
        self.style.configure('Card.TLabelframe', 
                           background='white',
                           borderwidth=1,
                           relief='solid',
                           bordercolor='#dee2e6')
        self.style.configure('Card.TLabelframe.Label', 
                           background='white',
                           foreground=self.primary_color,
                           font=('Segoe UI', 12, 'bold'))
        
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        
    def on_closing(self):
        """Handle window close event"""
        sys.stdout = sys.__stdout__ 
        self.root.destroy()
    
    def browse_file(self):
        file_path = filedialog.askopenfilename(
            title="Select XML File",
            filetypes=(("XML files", "*.xml"), ("All files", "*.*"))
        )
        if file_path:
            self.file_path.set(file_path)
    
    def start_analysis(self):
        journal = self.journal_var.get()
        if not journal:
            messagebox.showerror("No Journal Selected", "Please select a journal before starting analysis.")
            self.status_var.set("Error: Please select a journal first.")
            return
        file_path = self.file_path.get()
        if not file_path:
            self.status_var.set("Error: Please select a file first")
            return
        if not Path(file_path).exists():
            self.status_var.set(f"Error: File not found: {file_path}")
            return
        # Copy selected file to data/<journal>/input
        dest_dir = os.path.join('data', journal, 'input')
        os.makedirs(dest_dir, exist_ok=True)
        file_name = os.path.basename(file_path)
        dest_path = os.path.join(dest_dir, file_name)
        try:
            import shutil
            shutil.copyfile(file_path, dest_path)
            file_path_for_analysis = dest_path
        except Exception as e:
            self.status_var.set(f"Error copying file to journal input dir: {e}")
            return
        self.analyze_btn.config(state=tk.DISABLED)
        self.browse_btn.config(state=tk.DISABLED)
        self.progress.start(10)
        self.status_var.set("Analyzing... Please wait")
        self.clear_output()
        analysis_thread = threading.Thread(
            target=self.run_analysis_thread,
            args=(file_path_for_analysis, journal),
            daemon=True
        )
        analysis_thread.start()
        self.check_thread_status(analysis_thread)

    def run_analysis_thread(self, file_path, journal):
        try:
            asyncio.run(self.run_pipeline_with_file(file_path, journal))
        except Exception as e:
            print(f"Error during analysis: {str(e)}")

    async def run_pipeline_with_file(self, file_path, journal):
        print(f"Starting analysis of: {file_path}, journal: {journal}")
        print("-" * 50)
        original_argv = sys.argv.copy()
        try:
            from main import run_pipeline
            output = await run_pipeline(analyzer=None, file_path=file_path, journal=journal)
            if asyncio.iscoroutine(output):
                output = await output
            if output:
                self.root.after(0, lambda: self.display_json_results(output))
            return output
        except Exception as e:
            error_msg = f"Error: {str(e)}"
            print(error_msg)
            return {"error": error_msg}
        finally:
            sys.argv = original_argv
            print("\n" + "=" * 50)
            print("Analysis complete!")    

    def get_journal_folders(self):
        journal_path = os.path.join("data")
        try:
            if not os.path.isdir(journal_path):
                return []
            return [name for name in os.listdir(journal_path) if os.path.isdir(os.path.join(journal_path, name))]
        except Exception:
            return []

    def on_journal_selected(self):
        selected = self.journal_var.get()
        if not selected:
            self.browse_btn.config(state=tk.DISABLED)
            self.analyze_btn.config(state=tk.DISABLED)
            self.status_var.set("Please select a journal before analysis.")
        else:
            self.browse_btn.config(state=tk.NORMAL)
            self.analyze_btn.config(state=tk.NORMAL)
            self.status_var.set(f"Ready: Journal '{selected}' selected.")    
    
    def check_thread_status(self, thread):
        if thread.is_alive():
            self.root.after(100, lambda: self.check_thread_status(thread))
        else:
            self.progress.stop()
            self.analyze_btn.config(state=tk.NORMAL)
            self.browse_btn.config(state=tk.NORMAL)
            self.status_var.set("Analysis completed")
    
    def display_json_results(self, json_data):
        """Display JSON data in a formatted way in the JSON tab"""
        try:
            if isinstance(json_data, str) and (json_data.startswith('{') or json_data.startswith('[')):
                try:
                    json_data = json.loads(json_data)
                except json.JSONDecodeError:
                    pass
                    
            if isinstance(json_data, (dict, list)):
                json_str = json.dumps(json_data, indent=2, ensure_ascii=False)
            else:
                json_str = str(json_data)
                
            self.json_text.config(state=tk.NORMAL)
            self.json_text.delete(1.0, tk.END)
            
            self._format_json(json_str)
            
            self.notebook.select(0)
            
        except Exception as e:
            self.json_text.config(state=tk.NORMAL)
            self.json_text.delete(1.0, tk.END)
            self.json_text.insert(tk.END, f"Error formatting JSON: {str(e)}\n\nType: {type(json_data)}\nValue: {json_data!r}")
        finally:
            self.json_text.config(state=tk.DISABLED)
            self.json_text.see(tk.END)

    def _format_json(self, json_str):
        """Format JSON string with syntax highlighting"""
        in_string = False
        in_escape = False
        key = True
        i = 0
        
        while i < len(json_str):
            char = json_str[i]
            
            if char == '"' and not in_escape:
                if not in_string:
                    start = i
                    in_string = True
                    key = True
                else:
                    end = i + 1
                    if key:
                        self.json_text.insert(tk.END, json_str[start:end], "key")
                        key = False
                    else:
                        self.json_text.insert(tk.END, json_str[start:end], "string")
                    in_string = False
            
            elif in_string:
                i += 1
                continue
                
            elif char in ['{', '}', '[', ']', ',', ':']:
                self.json_text.insert(tk.END, char, "bracket")
                
            elif char.isdigit() or (char == '-' and i + 1 < len(json_str) and json_str[i+1].isdigit()):
                start = i
                if char == '-':
                    i += 1
                while i < len(json_str) and (json_str[i].isdigit() or json_str[i] in '.eE-+'):
                    i += 1
                self.json_text.insert(tk.END, json_str[start:i], "number")
                i -= 1
                
            elif json_str.startswith("true", i) or json_str.startswith("false", i):
                end = i + (4 if json_str[i] == 't' else 5)
                self.json_text.insert(tk.END, json_str[i:end], "boolean")
                i = end - 1
                
            elif json_str.startswith("null", i):
                self.json_text.insert(tk.END, "null", "null")
                i += 3
                
            else:
                self.json_text.insert(tk.END, char)
                
            i += 1
            
            if in_string and json_str[i-1] == '\\':
                in_escape = not in_escape
            else:
                in_escape = False
    
    def clear_output(self):
        """Clear both console and JSON outputs"""
        # Clear console
        self.output_text.config(state=tk.NORMAL)
        self.output_text.delete(1.0, tk.END)
        self.output_text.config(state=tk.DISABLED)
        
        # Clear JSON
        self.json_text.config(state=tk.NORMAL)
        self.json_text.delete(1.0, tk.END)
        self.json_text.config(state=tk.DISABLED)
        
        # Switch back to console tab
        self.notebook.select(0)
        self.status_var.set("Output cleared")

class TextRedirector:
    def __init__(self, widget, tag="stdout"):
        self.widget = widget
        self.tag = tag
    
    def write(self, text):
        self.widget.config(state=tk.NORMAL)
        self.widget.insert(tk.END, text, (self.tag,))
        self.widget.see(tk.END)
        self.widget.config(state=tk.DISABLED)
        self.widget.update_idletasks()
    
    def flush(self):
        pass

if __name__ == "__main__":
    root = tk.Tk()
    app = XMLAnalyzerApp(root)
    root.mainloop()
