# Integrating YOLOv8
## Nodes
- **hf_node: HFNode** -- This node accepts multiple video node and applies specified Hugging Face models on frames of those videos. Need to specify specific model in the configuration file. Currently testing with models with computer vision tasks.
- **hf_display: HFDisplay** -- This display the results alongside the source video.


**Using HF models require installation of additional packages:**
- transformers: HF library for transformer models for various tasks

