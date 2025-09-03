import gradio as gr
import os
import shutil
import json
import asyncio
from pathlib import Path

# Import your async analysis logic
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


async def analyze_file_async(journal, file_path):
    """Async handler to process uploaded XML file."""
    logs = []
    if not journal:
        return "Error: No journal selected", "{}"
    if not file_path:
        return "Error: No file uploaded", "{}"

    try:
        # Prepare destination
        dest_dir = os.path.join("data", journal, "input")
        os.makedirs(dest_dir, exist_ok=True)

        file_name = os.path.basename(file_path)
        dest_path = os.path.join(dest_dir, file_name)

        # Copy file
        shutil.copy(file_path, dest_path)
        logs.append(f"Starting analysis of: {dest_path}, journal: {journal}")
        logs.append("-" * 50)

        # Run pipeline
        output = await run_pipeline(file_path=dest_path, journal=journal)
        if asyncio.iscoroutine(output):
            output = await output

        logs.append("=" * 50)
        logs.append("Analysis complete!")

        # Format JSON
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

        return "\n".join(logs), json_output

    except Exception as e:
        logs.append(f"Error: {str(e)}")
        return "\n".join(logs), "{}"


def analyze_file(journal, file_path):
    """Sync wrapper to run async analysis in Gradio."""
    return asyncio.run(analyze_file_async(journal, file_path))


def gradio_ui():
    journals = get_journal_folders()

    with gr.Blocks(theme=gr.themes.Soft()) as demo:
        gr.Markdown(
            """
            # 🧩 XML Change Predictor
            AI-powered suggestions for smarter XML evolution.

            Upload an XML file to discover likely changes, frequent patterns, and improvement ideas — all powered by data-driven AI.
            """
        )

        with gr.Row():
            journal_dropdown = gr.Dropdown(
                choices=journals,
                label="Select Journal",
                info="Select a journal folder from data/",
                interactive=True,
            )

        with gr.Row():
            file_upload = gr.File(
                file_types=[".xml"],
                label="Upload XML File",
                type="filepath"
            )

        analyze_btn = gr.Button("Start Analysis", variant="primary")

        with gr.Tab("Console Output"):
            console_output = gr.Textbox(
                label="Console Logs",
                placeholder="Logs will appear here...",
                lines=15
            )

        with gr.Tab("JSON Results"):
            json_output = gr.Code(
                label="JSON Results",
                language="json"
            )

        analyze_btn.click(
            fn=analyze_file,
            inputs=[journal_dropdown, file_upload],
            outputs=[console_output, json_output]
        )

    return demo


if __name__ == "__main__":
    app = gradio_ui()
    app.launch()
