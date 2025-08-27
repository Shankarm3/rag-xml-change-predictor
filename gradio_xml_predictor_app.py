import gradio as gr
import json
from pathlib import Path
from main import ChangeAnalyzer, generate_change_prediction, extract_and_save_diffs

def load_analyzer():
    diffs_file = Path("processed/diffs.jsonl")
    analyzer = ChangeAnalyzer()
    if not diffs_file.exists():
        data_v1, data_v2 = Path("data/v1"), Path("data/v2")
        if data_v1.exists() and data_v2.exists():
            analyzer = extract_and_save_diffs(str(data_v1), str(data_v2), str(diffs_file))
        else:
            return None, "Diffs not found. Please run extraction or provide processed/diffs.jsonl."
    else:
        with open(diffs_file, "r", encoding="utf-8") as f:
            for line in f:
                item = json.loads(line)
                diff = item.get("diff", [])
                analyzer.analyze_diff(diff, '', '')
    return analyzer, None

def predict_changes(xml_file):
    analyzer, err = load_analyzer()
    if err:
        return gr.update(value=f"Error: {err}", visible=True), gr.update(visible=False), gr.update(visible=False), gr.State(None), gr.update(interactive=False)
    try:
        if hasattr(xml_file, 'read'):
            xml_content = xml_file.read()
            try:
                xml_content = xml_content.decode("utf-8")
            except UnicodeDecodeError:
                xml_content = xml_content.decode("latin-1")
        elif isinstance(xml_file, str) and Path(xml_file).exists():
            with open(xml_file, 'r', encoding='utf-8') as f:
                xml_content = f.read()
        else:
            xml_content = xml_file
    except Exception as e:
        return gr.update(value=f"Failed to read file: {e}", visible=True), gr.update(visible=False), gr.update(visible=False), gr.State(None), gr.update(interactive=False)
    result = generate_change_prediction(analyzer, xml_content)
    if 'error' in result:
        preview = xml_content[:200] if xml_content else '[EMPTY CONTENT]'
        return gr.update(value=f"Error: {result['error']}\n\nFile preview: {preview}", visible=True), gr.update(visible=False), gr.update(visible=False), gr.State(None), gr.update(interactive=False)
    suggested = result.get('suggested_changes', [])
    improvements = result.get('potential_improvements', [])
    patterns = result.get('change_patterns', [])
    suggested_str = "\n\n".join([
        f"[{i+1}] {item['xpath']} ({item['tag']})\n" +
        f"Current: {item['current_value']}\n" +
        f"Suggest: {item['suggested_change']['from']} -> {item['suggested_change']['to']} (confidence {item['suggested_change']['confidence']}%)\nPattern: {item['suggested_change']['pattern']} (occurrences {item['suggested_change']['occurrences']})"
        for i, item in enumerate(suggested)
    ]) or "No specific suggested changes found."
    improvements_str = "\n\n".join([
        f"[{i+1}] {item['xpath']} ({item['tag']})\nCurrent: {item['current_value']}\nSuggestion: {item['suggestion']}\nConfidence: {item['confidence']}%"
        for i, item in enumerate(improvements)
    ]) or "No general improvement suggestions."
    patterns_str = "\n".join([f"- {p['pattern']} (x{p['frequency']})" for p in patterns]) or "No frequent change patterns."
    return (
        gr.update(value=suggested_str, visible=True),
        gr.update(value=improvements_str, visible=True),
        gr.update(value=patterns_str, visible=True),
        gr.State(result),
        gr.update(interactive=True)
    )

def show_json(result):
    value = getattr(result, 'value', result)
    if not value:
        return gr.update(value="No result.", visible=True)
    try:
        return gr.update(
            value=json.dumps(value, indent=2, default=str),
            visible=True
        )
    except Exception as e:
        return gr.update(value=f"Unable to display JSON: {e}", visible=True)

def on_file_upload(xml_file):
    if xml_file:
        # Analyze enabled, don't clear outputs
        return (
            gr.update(interactive=True),                   # Analyze button
            gr.update(value="", visible=False),          # Suggested Changes
            gr.update(value="", visible=False),          # Improvement Suggestions
            gr.update(value="", visible=False),          # Frequent Change Patterns
            gr.State(None),                               # JSON state
            gr.update(interactive=False),                 # Show JSON button disabled
            gr.update(value="", visible=False)           # Raw JSON box hidden
        )
    else:
        # File cleared: disable, clear everything
        return (
            gr.update(interactive=False),                 # Analyze button
            gr.update(value="", visible=False),          # Suggested Changes
            gr.update(value="", visible=False),          # Improvement Suggestions
            gr.update(value="", visible=False),          # Frequent Change Patterns
            gr.State(None),                               # JSON state
            gr.update(interactive=False),                 # Show JSON button disabled
            gr.update(value="", visible=False)           # Raw JSON box hidden
        )

def build_ui():
    custom_css = """
    .gradio-container {
        background-image: linear-gradient(to right, #314755 0%, #26a0da  51%, #314755  100%)!important;
        min-height: 100vh;
        padding-bottom: 40px;
    }
    .hero-card {
        max-width:840px; margin:2.5em auto 2.2em auto; border-radius:28px;
        background:linear-gradient(105deg, #fff 62%, #e0fcff 128%);
        box-shadow: 0 4px 40px #d7eaff88, 0 12px 60px #aad4ee13;
        padding: 1.8em 2.8em;
    }
    .demo-title {
        font-size:2.65rem;
        font-weight:900;
        text-align:center;
        margin-bottom:.35em;
        color:#0280ff;
        letter-spacing: .04em;
        text-shadow: 1.5px 1.2px 10px #4fc3f366, 0 2px 0 #fff;
    }
    .blurb {
        text-align:center;
        color:#314870;
        font-size: 1.22rem;
        line-height:1.48;
        padding:.6em 0 .15em 0;
    }
    .analyze-btn {
        background:linear-gradient(90deg,#6a11cb  8%,#2575fc 92%)!important;
        color:#fff!important;
        border-radius: 32px!important;
        font-size:1.16rem!important;
        font-weight:900!important;
        box-shadow:0 4px 12px #1466b616, 0 2px 1.7px #34fadc21;
        border:none!important;
        letter-spacing:.055em;
        padding:.85em 2.3em!important;
        margin-right:0.7em;
        transition: background 0.17s, transform 0.17s, box-shadow 0.13s;
    }
    .analyze-btn:hover:enabled {
        background:linear-gradient(90deg,#18c6d7 22%,#2186e5 84%)!important;
        box-shadow: 0 4px 24px #0ac7fa77;
        transform: translateY(-2px) scale(1.035);
    }
    .analyze-btn:disabled {filter: grayscale(0.65) brightness(1.29)!important;opacity: 0.55!important;}
    .json-btn {
        background:linear-gradient(90deg,#ffb347 0,#ffcc33 100%)!important;
        color:#452c00!important;
        border-radius:32px!important;
        font-size:1.08rem!important;
        font-weight:900!important;
        letter-spacing:.03em;
        border:none!important;
        padding:.78em 2.1em!important;
        transition: background 0.2s, transform 0.15s;
    }
    .json-btn:enabled:hover {
        background:linear-gradient(90deg,#ffd76d 5%,#ffdf8d 95%)!important;
        color:#7c5400!important;
        transform:translateY(-2px) scale(1.045);
        box-shadow:0 3px 16px #ffe5675c;
    }
    .json-btn:disabled {filter: grayscale(0.6) brightness(.98)!important;opacity: 0.6!important;}
    .card-result {
        padding:1.35em 1.6em 1.33em 1.6em;
        background:linear-gradient(102deg,#ecfdff 80%,#f6f2fd 128%);
        border-radius:16px;
        box-shadow: 0 3px 20px #30a3f637, 0 4px 30px #e1edfa1f;
        margin-bottom: 1.44em;
        border-left: 6px solid #70c1ff;
    }
    .card-result textarea {
        background: #fafdff;
        font-weight: 500;
        color: #0a2334;
    }
    .gr-box { box-shadow:0 2px 17px #dedffe39!important; }
    .gr-accordion {border-radius: 14px!important; margin-bottom: 18px;}
    .gr-accordion .gr-panel {background: #fbfeff!important;}
    .gr-code code {background: #f6fdff!important;font-family: "Fira Mono","Menlo",monospace;font-size:.99rem;border-radius:7px; color:#13497f;}
    label[for^='component'] { font-weight:700; color:#2a6699; font-size: 1.13rem; letter-spacing: .01em; }
    """
    with gr.Blocks(css=custom_css, title="XML Change Predictor Demo") as demo:
        gr.HTML("""
        <div class='hero-card'>
            <div class='demo-title'>🧩 XML Change Predictor</div>
            <div class='blurb'>
                Upload a new XML file.<br>
                Suggests changes & patterns learned between your v1/v2 XML data.<br>
                <span style='color:#1467b8;font-weight:800'>AI-powered analysis for smarter XML evolution!</span>
            </div>
        </div>
        """)
        with gr.Row():
            file_in = gr.File(label="Upload XML for Prediction", file_types=[".xml"], interactive=True)
        with gr.Row():
            analyze_btn = gr.Button("🔍 Analyze", interactive=False, elem_classes=["analyze-btn"])
            json_btn = gr.Button("📄 Show JSON", interactive=False, elem_classes=["json-btn"])
        with gr.Row():
            with gr.Column():
                out_suggested = gr.Textbox(label="Suggested Changes", visible=False, lines=10, interactive=False, elem_classes=["card-result"])
            with gr.Column():
                out_impr = gr.Textbox(label="Improvement Suggestions", visible=False, lines=8, interactive=False, elem_classes=["card-result"])
        with gr.Accordion("🌐 See most frequent change patterns...", open=False):
            out_patterns = gr.Textbox(label="Frequent Change Patterns", visible=False, lines=8, interactive=False, elem_classes=["card-result"])
        with gr.Accordion("📂 Show result as JSON", open=False) as acc_json:
            out_json = gr.Code(label="Raw JSON", visible=False, language="json")
        state = gr.State(value=None)

        file_in.change(
            on_file_upload,
            inputs=file_in,
            outputs=[
                analyze_btn,            # (0) Enable/disable Analyze button
                out_suggested,         # (1) Clear/Hide Suggested
                out_impr,              # (2) Clear/Hide Improvements
                out_patterns,          # (3) Clear/Hide Patterns
                state,                 # (4) Reset state
                json_btn,              # (5) Disable Show JSON
                out_json               # (6) Hide JSON output
            ]
        )
        analyze_btn.click(predict_changes, inputs=[file_in], outputs=[out_suggested, out_impr, out_patterns, state, json_btn])
        json_btn.click(show_json, inputs=state, outputs=out_json)
    return demo

if __name__ == "__main__":
    demo = build_ui()
    demo.launch(share=True)
