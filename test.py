import gradio as gr

def f(journal, xml):
    return f"Journal: {journal}, File: {xml.name if hasattr(xml, 'name') else xml}"

gr.Interface(
    fn=f,
    inputs=[gr.Dropdown(['j1', 'j2']), gr.File(type='filepath')],
    outputs="text"
).launch()