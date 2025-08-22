import os
import streamlit as st
import tempfile
import json
from main import run_pipeline
import asyncio

# Page config
st.set_page_config(
    page_title="XML Change Predictor",
    page_icon="🔍",
    layout="wide"
)

# Custom CSS for better UI
st.markdown("""
    <style>
    .main-header { font-size: 2.5rem; color: #1f77b4; text-align: center; margin-bottom: 1rem; }
    .result-section { margin-top: 2rem; padding: 1rem; border-radius: 0.5rem; background-color: #f8f9fa; }
    .file-info { font-size: 1.1rem; margin-bottom: 1rem; }
    .stButton>button { width: 100%; padding: 0.5rem; font-size: 1.1rem; }
    .stProgress > div > div > div > div { background-color: #1f77b4; }
    </style>
""", unsafe_allow_html=True)

# App header
st.markdown('<p class="main-header">🔍 XML Change Predictor</p>', unsafe_allow_html=True)

# File upload section
uploaded_file = st.file_uploader("Upload an XML file for analysis", type=["xml"])

# Initialize session state
if 'analysis_done' not in st.session_state:
    st.session_state.analysis_done = False
if 'results' not in st.session_state:
    st.session_state.results = None

# Process button and analysis
if uploaded_file is not None:
    with tempfile.TemporaryDirectory() as temp_dir:
        # Save uploaded file to temp directory
        temp_file_path = os.path.join(temp_dir, uploaded_file.name)
        with open(temp_file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        # Ensure input directory exists
        os.makedirs("data/input", exist_ok=True)
        
        # Copy file to input directory for processing
        input_file_path = os.path.join("data/input", uploaded_file.name)
        with open(input_file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        # Display file info
        st.markdown(f'<div class="file-info">📄 File uploaded: <strong>{uploaded_file.name}</strong> ({uploaded_file.size} bytes)</div>', 
                   unsafe_allow_html=True)
        
        # Analyze button
        if st.button("Analyze XML File"):
            with st.spinner("Analyzing file. This may take a few minutes..."):
                try:
                    # Create a progress bar
                    progress_bar = st.progress(0)
                    
                    # Run the analysis pipeline
                    progress_bar.progress(20, text="Initializing analysis...")
                    
                    # Run the async pipeline
                    async def run_analysis():
                        progress_bar.progress(40, text="Processing XML file...")
                        await run_pipeline()
                        
                        # Read the generated results
                        output_file = os.path.join("predictions", f"suggestions_{uploaded_file.name}.json")
                        if os.path.exists(output_file):
                            with open(output_file, 'r') as f:
                                st.session_state.results = json.load(f)
                            st.session_state.analysis_done = True
                            progress_bar.progress(100, text="Analysis complete!")
                        else:
                            st.error("Analysis completed but no results were generated.")
                    
                    # Run the async function
                    asyncio.run(run_analysis())
                    
                except Exception as e:
                    st.error(f"An error occurred during analysis: {str(e)}")
                    progress_bar.empty()

# Display results if analysis is done
if st.session_state.analysis_done and st.session_state.results:
    st.markdown("## Analysis Results")
    
    # Display suggested changes
    if st.session_state.results.get('analyzer_predictions', {}).get('suggested_changes'):
        st.markdown("### Suggested Changes")
        for i, change in enumerate(st.session_state.results['analyzer_predictions']['suggested_changes'], 1):
            with st.expander(f"{i}. {change['suggested_change']['from']} → {change['suggested_change']['to']} (Confidence: {change['suggested_change']['confidence']}%)"):
                st.code(f"""
Current value: {change['current_value']}
XPath: {change['xpath']}
Confidence: {change['suggested_change']['confidence']}%
Pattern: {change['suggested_change']['pattern']}
Occurrences: {change['suggested_change']['occurrences']}
                """.strip(), language='text')
    
    # Display potential improvements
    if st.session_state.results.get('analyzer_predictions', {}).get('potential_improvements'):
        st.markdown("### Potential Improvements")
        for i, imp in enumerate(st.session_state.results['analyzer_predictions']['potential_improvements'], 1):
            with st.expander(f"{i}. {imp['tag']} (Confidence: {imp['confidence']}%)"):
                st.code(f"""
Current: {imp['current_value']}
XPath: {imp['xpath']}
Suggestion: {imp['suggestion']}
Change count: {imp.get('change_count', 'N/A')}
                """.strip(), language='text')
    
    # Display raw JSON in an expander
    with st.expander("View Raw JSON Output"):
        st.json(st.session_state.results)
