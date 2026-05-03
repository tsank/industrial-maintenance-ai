"""
ui/chatbot_tab.py

Tab 1: Clean chatbot interface for end users.
Calls the run() public API and displays the answer.
No trace information is shown here.
"""

import gradio as gr
from compressor_observability import run


def _respond(message: str, history: list) -> str:
    """Called by Gradio on each user message."""
    try:
        result = run(query=message)
        return result.answer if result.answer else "No answer returned."
    except Exception as e:
        return f"Error: {str(e)}"


def build_chatbot_tab() -> gr.ChatInterface:
    return gr.ChatInterface(
        fn=_respond,
        title="Industrial Maintenance AI",
        description=(
            "Ask questions about the Atlas Copco GA5 air compressor. "
            "Answers are retrieved from the maintenance manual using "
            "multi-store agentic RAG with long-term memory."
        ),
        examples=[
            "What is the maximum working pressure of the GA5?",
            "How do I modify the unloading pressure setting?",
            "What components are involved in a motor overload shutdown?",
            "What are the service intervals and which plans apply?",
        ],
    )
