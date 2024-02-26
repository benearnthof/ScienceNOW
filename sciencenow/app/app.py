"""
Gradio app to deploy the model.
"""

import gradio as gr
import datetime

# import config and build a user interface with inputs corresponding to the types
from sciencenow.config import (
    setup_params, 
    online_params,
)

def predict(*args):
    return args

# build interface 
with gr.Blocks() as demo:
    gr.Markdown("Set all Model Parameters & click 'Submit'.")
    with gr.Row():
        with gr.Column():
            with gr.Tabs():
                with gr.TabItem("Primary Data"):
                    with gr.Row():
                        startdate = gr.Textbox(value="01 01 2020", label="Start Date of Period", placeholder="Format: Day Month Year")
                        enddate = gr.Textbox(value="31 01 2020", label="Start Date of Period", placeholder="Format: Day Month Year")
                    with gr.Row():
                        hdbscan = gr.Slider(minimum=1, maximum=200, value=25, label="HDBSCAN Cluster Size")
                    with gr.Row():
                        target = gr.Textbox(value="cs", label="Target Class")
                    with gr.Row():
                        recompute = gr.Checkbox(value=False, label="Recompute UMAP?")
                    with gr.Row():
                        bins = gr.Slider(minimum=1, maximum=52, value=4, label="Number of Temporal Bins", step=1)
                    with gr.Row():
                        limit = gr.Slider(minimum=1, maximum=10000, value=5000, label="Maximum # of Papers", step=1)
                with gr.TabItem("Synthetic Data"):
                    with gr.Row():
                        secondary_startdate = gr.Textbox(value="01 01 2020", label="Secondary Start Date", placeholder="Format: Day Month Year")
                        secondary_enddate = gr.Textbox(value="31 12 2020", label="Secondary End Date", placeholder="Format: Day Month Year")
                    with gr.Row():
                        secondary_target = gr.Textbox(value="q-bio", label="Secondary Target Class")
                    with gr.Row():
                        secondary_proportion = gr.Number(value=0.1, label="Proportion of Synthetic/Real", minimum=0, maximum=1)
                    with gr.Row():
                        n_trends = gr.Slider(minimum=1, maximum=4, label="Number of Synthetic Trends", step=1)
                    with gr.Row():
                        trend_deviation = gr.Number(value=1.5, label="Deviation of Synthetic Trend", minimum=1, maximum=3)
                submit_button = gr.Button("Submit")
        with gr.Column():
            output_placeholder=gr.Textbox()
    submit_button.click(predict, inputs=[
        startdate,
        enddate,
        hdbscan,
        target,
        recompute,
        bins,
        limit,
        secondary_startdate,
        secondary_enddate,
        secondary_target,
        secondary_proportion,
        n_trends,
        trend_deviation
        ], outputs=output_placeholder)


demo.launch()
