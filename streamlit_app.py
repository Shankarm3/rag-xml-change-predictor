import streamlit as st
import json
from pathlib import Path
from main import ChangeAnalyzer, extract_and_save_diffs, generate_change_prediction

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

def run_app():
    st.set_page_config(page_title="XML Change Predictor", layout="centered")
    st.markdown(
        """
        <div style='margin-bottom: 1.5em; padding: 1.2em; border-radius:20px; background: linear-gradient(92deg,#e0fcff 70%,#c8e3fc 120%); box-shadow: 0 2px 20px #30a3f633;'>
        <h1 style='text-align:center; color:#0280ff; font-weight:900; margin-bottom:.18em;'>🧩 XML Change Predictor</h1>
        <div style='text-align:center; color:#314870; font-size:1.22rem;'>
        Upload a new XML file.<br>
        Suggests changes & patterns learned between your v1/v2 XML data.<br>
        <span style='color:#1467b8;font-weight:800'>AI-powered analysis for smarter XML evolution!</span>
        </div></div>
        """,
        unsafe_allow_html=True,
    )

    uploaded_file = st.file_uploader("Upload XML for Prediction", type=["xml"])

    run_analyze = st.button("🔍 Analyze", disabled=not uploaded_file)

    if run_analyze and uploaded_file:
        analyzer, err = load_analyzer()
        if err:
            st.error(err)
            return
        try:
            xml_content = uploaded_file.read()
            try:
                xml_content = xml_content.decode("utf-8")
            except UnicodeDecodeError:
                xml_content = xml_content.decode("latin-1")
        except Exception as e:
            st.error(f"Failed to read file: {e}")
            return
        result = generate_change_prediction(analyzer, xml_content)
        if 'error' in result:
            preview = xml_content[:200] if xml_content else '[EMPTY CONTENT]'
            st.error(f"Error: {result['error']}\n\nFile preview: {preview}")
            return
        suggested = result.get('suggested_changes', [])
        improvements = result.get('potential_improvements', [])
        patterns = result.get('change_patterns', [])

        st.subheader("Suggested Changes")
        if suggested:
            for i, item in enumerate(suggested):
                st.markdown(f"**[{i+1}] {item['xpath']} ({item['tag']})**")
                st.markdown(f"- Current: `{item['current_value']}`\n- Suggest: `{item['suggested_change']['from']}` → `{item['suggested_change']['to']}` (confidence {item['suggested_change']['confidence']}%)\n- Pattern: `{item['suggested_change']['pattern']}` (occurrences {item['suggested_change']['occurrences']})")
        else:
            st.info("No specific suggested changes found.")

        st.subheader("Improvement Suggestions")
        if improvements:
            for i, item in enumerate(improvements):
                st.markdown(f"**[{i+1}] {item['xpath']} ({item['tag']})**")
                st.markdown(f"- Current: `{item['current_value']}`\n- Suggestion: {item['suggestion']}\n- Confidence: {item['confidence']}%")
        else:
            st.info("No general improvement suggestions.")

        with st.expander("🌐 See most frequent change patterns..."):
            if patterns:
                for p in patterns:
                    st.markdown(f"- {p['pattern']} (x{p['frequency']})")
            else:
                st.write("No frequent change patterns.")

        with st.expander("📂 Show result as JSON"):
            st.code(json.dumps(result, indent=2, default=str), language="json")

if __name__ == "__main__":
    run_app()
