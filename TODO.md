# TODO: Create Gradio UI for XML Change Predictor

## Completed Tasks
- [x] Add gradio>=4.0.0 to requirements.txt
- [x] Create gradio_app.py with UI components:
  - Journal selection dropdown
  - XML file upload
  - Analysis button
  - Output tabs for suggested changes, improvements, patterns, and JSON
  - Status display
  - Error handling

## Pending Tasks
- [ ] Test the Gradio app locally
- [ ] Verify integration with main.py pipeline
- [ ] Provide instructions for running the app
- [ ] Test with sample XML files
- [ ] Handle edge cases (no journals, invalid files, etc.)

## Instructions to Run
1. Install dependencies: `pip install -r requirements.txt`
2. Ensure data directories exist with journal folders (e.g., data/mnras/)
3. Run the app: `python gradio_app.py`
4. Open browser at http://localhost:7860

## Notes
- The UI follows similar functionality to tkinter_xml_rag.py
- Uses Gradio Blocks for layout
- Integrates with existing run_pipeline function from main.py
- Supports journal selection and file upload
- Displays results in organized tabs
