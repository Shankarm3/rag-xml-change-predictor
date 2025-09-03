import gradio as gr
import os
import io
import sys
import json
import asyncio
import shutil
import contextlib

from main import run_pipeline


def get_journal_folders():
    """Get available journals from data/ directory."""
    journal_path = "data"
    try:
        if not os.path.isdir(journal_path):
            return []
        return [
            name for name in os.listdir(journal_path)
            if os.path.isdir(os.path.join(journal_path, name))
        ]
    except Exception:
        return []


async def analyze_file_async(journal, file_obj):
    """Async version of analyze_file with full console log capture."""
    if not journal:
        return "Error: No journal selected", {}

    if not file_obj:
        return "Error: No file uploaded", {}

    dest_dir = os.path.join("data", journal, "input")
    os.makedirs(dest_dir, exist_ok=True)

    file_name = os.path.basename(file_obj.name)
    dest_path = os.path.join(dest_dir, file_name)
    try:
        shutil.copy(file_obj.name, dest_path)
    except Exception as e:
        return f"Error copying file: {str(e)}", {}

    log_buffer = io.StringIO()
    try:
        with contextlib.redirect_stdout(log_buffer):
            print(f"Starting analysis of: {dest_path}, journal: {journal}")
            print("-" * 50)

            output = await run_pipeline(analyzer=None, file_path=dest_path, journal=journal)
            if asyncio.iscoroutine(output):
                output = await output

            print("=" * 50)
            print("Analysis complete!")

        logs = log_buffer.getvalue()

        if isinstance(output, (dict, list)):
            json_output = json.dumps(output, indent=2, ensure_ascii=False)
        elif isinstance(output, str):
            try:
                parsed = json.loads(output)
                json_output = json.dumps(parsed, indent=2, ensure_ascii=False)
            except json.JSONDecodeError:
                json_output = output
        else:
            json_output = str(output)

        return logs, json_output

    except Exception as e:
        error_msg = f"Error: {str(e)}"
        logs = log_buffer.getvalue() + "\n" + error_msg
        return logs, {}


def analyze_file(journal, file_path):
    """Sync wrapper to run async analysis in Gradio."""
    return asyncio.run(analyze_file_async(journal, file_path))


def gradio_ui():
    with gr.Blocks(title="XML Change Predictor", theme=gr.themes.Soft()) as app:
        gr.Markdown("# XML Change Predictor")
        gr.Markdown("Upload an XML file to analyze potential changes based on historical patterns.")
        
        with gr.Row():
            # Journal selection (hidden for demo)
            # available_journals = get_journal_folders()
            # journal_dropdown = gr.Dropdown(
            #     label="Select Journal",
            #     choices=available_journals,
            #     interactive=bool(available_journals),
            #     value=available_journals[0] if available_journals else None
            # )
            
            # Default to 'mnras' for demo
            journal_dropdown = gr.Textbox(
                label="Journal",
                value="mnras",
                visible=False  # Hidden but still passed to the function
            )
            
            file_upload = gr.File(
                label="Upload XML File",
                file_types=[".xml"],
                type="filepath"
            )

        analyze_btn = gr.Button("Start Analysis", variant="primary")

        with gr.Tab("Console Logs"):
            console_output = gr.Textbox(
                label="Console Output",
                lines=15,
                interactive=False
            )

        with gr.Tab("JSON Results"):
            json_output = gr.Code(
                label="JSON Results",
                language="json",
                elem_classes=["json-output"]
            )

        analyze_btn.click(
            fn=analyze_file,
            inputs=[journal_dropdown, file_upload],
            outputs=[console_output, json_output],
            show_progress="full"
        )

    return app


if __name__ == "__main__":
    app = gradio_ui()
    app.launch()
