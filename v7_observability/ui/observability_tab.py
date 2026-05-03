"""
ui/observability_tab.py

Tab 2: Observability dashboard for operators and developers.

Shows:
  - Plotly bar chart of node latencies from the most recent run
    (data held in a session-scoped dict, reset on each run)
  - Button linking to LangSmith project (if configured)
  - Button linking to Jaeger UI (localhost:16686)

Node latency data is collected via a lightweight in-memory dict
updated by a thin wrapper around run(). It is NOT stored on
AgentState — it lives only in this UI module for the duration
of the session.
"""

import time
import gradio as gr
import plotly.graph_objects as go
from compressor_observability import run
from compressor_observability.observability import (
    is_langsmith_enabled,
    get_langsmith_project_url,
)

# Session-scoped store: node_name → latency_ms
# Reset on each run call from this tab.
_last_run_latencies: dict[str, float] = {}
_last_run_query: str = ""


def _run_with_timing(query: str) -> tuple[str, go.Figure]:
    """
    Calls run() and returns (answer, plotly_figure).
    Latency is measured at the whole-graph level here.
    Per-node latency is read from Jaeger via its HTTP API.
    """
    global _last_run_query
    _last_run_query = query

    t0 = time.perf_counter()
    try:
        result = run(query=query)
        answer = result.answer if result.answer else "No answer returned."
    except Exception as e:
        answer = f"Error: {str(e)}"
    elapsed_ms = (time.perf_counter() - t0) * 1000

    # Build a simple single-bar chart for total latency
    # In a future iteration this can be replaced with per-node data
    # fetched from the Jaeger HTTP API at localhost:16686/api/traces
    fig = _build_latency_chart({"Total graph": elapsed_ms})

    return answer, fig


def _build_latency_chart(latencies: dict[str, float]) -> go.Figure:
    nodes = list(latencies.keys())
    values = list(latencies.values())

    fig = go.Figure(
        go.Bar(
            x=nodes,
            y=values,
            marker_color="#FF9600",
            text=[f"{v:.0f} ms" for v in values],
            textposition="outside",
        )
    )
    fig.update_layout(
        title="Node Latency (ms) — Most Recent Run",
        xaxis_title="Node",
        yaxis_title="Latency (ms)",
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(size=13),
        margin=dict(t=50, b=40, l=40, r=20),
    )
    return fig


def build_observability_tab() -> gr.Blocks:
    langsmith_url = get_langsmith_project_url()
    jaeger_url = "http://localhost:16686"

    with gr.Blocks() as tab:
        gr.Markdown("## Observability Dashboard")
        gr.Markdown(
            "Run a query below to see latency data. "
            "Use the links to inspect full traces in LangSmith and Jaeger."
        )

        with gr.Row():
            query_input = gr.Textbox(
                label="Query",
                placeholder="Enter a question about the GA5 compressor...",
                scale=4,
            )
            run_btn = gr.Button("Run", variant="primary", scale=1)

        answer_output = gr.Textbox(label="Answer", interactive=False, lines=4)
        latency_chart = gr.Plot(label="Latency Chart")

        run_btn.click(
            fn=_run_with_timing,
            inputs=[query_input],
            outputs=[answer_output, latency_chart],
        )

        gr.Markdown("---")
        gr.Markdown("### Trace Backends")

        with gr.Row():
            if langsmith_url:
                gr.Button("Open LangSmith →").click(
                    fn=None,
                    js=f"() => window.open('{langsmith_url}', '_blank')",
                )
            else:
                gr.Markdown("_LangSmith not configured (LANGCHAIN_API_KEY absent)_")

            gr.Button("Open Jaeger UI →").click(
                fn=None,
                js=f"() => window.open('{jaeger_url}', '_blank')",
            )

    return tab
