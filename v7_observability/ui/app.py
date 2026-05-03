"""
ui/app.py

Gradio application entry point.
Launches a two-tab interface:

  Tab 1 — Chatbot       : clean Q&A for end users
  Tab 2 — Observability : latency chart, LangSmith link, Jaeger link

Run from the v7 root:
    python ui/app.py
"""

import gradio as gr
from ui.chatbot_tab import build_chatbot_tab
from ui.observability_tab import build_observability_tab

with gr.Blocks(title="Industrial Maintenance AI — v7") as demo:
    gr.Markdown(
        "# Industrial Maintenance AI\n"
        "Atlas Copco GA5 Compressor · Agentic RAG · Long-term Memory · Observability"
    )

    with gr.Tabs():
        with gr.Tab("Chatbot"):
            build_chatbot_tab()

        with gr.Tab("Observability"):
            build_observability_tab()

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)
