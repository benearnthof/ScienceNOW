"""
Gradio app to deploy the model.
"""

import gradio as gr
from typing import List, Dict, Any
from sciencenow.models.train import ModelWrapper

# import config and build a user interface with inputs corresponding to the types
from sciencenow.config import (
    setup_params, 
)

setup_dict = setup_params.copy()

def parse_inputs(args: List[Any], s_dict: Dict=setup_dict) -> Dict:
    """
    Helper function that takes in a list of arguments and returns a dictionary that matches
    `sciencenow.config.setup_params`
    """
    s_dict["startdate"]=args[0]
    s_dict["enddate"]= args[1]
    s_dict["cluster_size"]= args[2]
    s_dict["target"]= args[3]
    s_dict["recompute"]= args[4]
    s_dict["nr_bins"]= args[5]
    s_dict["limit"]= args[6]
    s_dict["threshold"] =args[7]
    s_dict["secondary_startdate"] = args[8]
    s_dict["secondary_enddate"]= args[9]
    if args[9] == "None" or args[10] == "":
        s_dict["secondary_target"]= None
    else:
        s_dict["secondary_target"]= args[10]
    s_dict["secondary_proportion"]= args[11]
    s_dict["n_trends"]= args[12]
    s_dict["trend_deviation"]= args[13]
    return s_dict

wrapper = 0

def wrapper_setup(setup_dict: Dict):
    """
    Setup function that creates wrapper if it does not exist yet and when called again will only 
    modify the subsets so we don't need to repeadedly load the arxiv df.
    """
    global wrapper
    if wrapper == 0:
        wrapper = ModelWrapper(setup_params=setup_dict, model_type="semisupervised")
    elif isinstance(wrapper, ModelWrapper):
        wrapper._reinitialize(setup_params=setup_dict)
    return wrapper

# wrapper_setup(setup_dict)

# Predictor class that instantiates a wrapper once and only loads the snapshot once to save time
def predict(*args):
    """
    Wrapper that merges setup parameters input by the user with default parameters that are necessary.
    Then Instantiates a Topic Model
    Then trains topic model and outputs results
    """
    setup_dict = parse_inputs(args)
    wrapper = wrapper_setup(setup_dict)
    return wrapper.setup_params

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
                        recompute = gr.Checkbox(value=False, label="Recompute UMAP?")
                    with gr.Row():
                        bins = gr.Slider(minimum=1, maximum=52, value=4, label="Number of Temporal Bins", step=1)
                    with gr.Row():
                        limit = gr.Slider(minimum=1, maximum=10000, value=5000, label="Maximum # of Papers", step=1)
                        threshold = gr.Slider(minimum=0, maximum=100, value=10, label="Minimum # of Papers in Labelclass", step=1)
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
    submit_button.click(
            predict, 
            inputs =
            [
                startdate,
                enddate,
                hdbscan,
                target,
                recompute,
                bins,
                limit,
                threshold,
                secondary_startdate,
                secondary_enddate,
                secondary_target,
                secondary_proportion,
                n_trends,
                trend_deviation
            ], 
            outputs=output_placeholder
        )


demo.launch()

# Next steps: 
# Run Model with input parameters
# extract outputs
# visualize in 3d scatter plot
# https://www.gradio.app/docs/plot
# https://github.com/MaartenGr/BERTopic/blob/257c4b5c0a80606acefc57339e713556dd94a1e6/bertopic/plotting/_topics.py#L11
# https://plotly.github.io/plotly.py-docs/generated/plotly.graph_objects.Scatter3d.html
# https://plotly.com/python/3d-scatter-plots/
