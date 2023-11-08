# Integrating YOLOv8
## Nodes
- **hf_text_node: HFTextNode** -- This node accepts textual input (right now supplied with hf_text) and applies specified Hugging Face models on input text. Need to specify specific model/task in the configuration file. 
- **hf_cv_node: HFCVNode** -- This node accepts input frames(right now supplied with hf_video) and applies specified Hugging Face CV models on input frames. Need to specify specific model/task in the configuration file. 

- **hf_vqa: HFVQANode** -- This node accepts input frames & question(right now supplied with data_vqa) and applies specified VQA models on input frames. Need to specify specific model/task in the configuration file. 

* Right now all the outputs are to command line...

## Example Use
- Example configs for all three nodes are in configs/huggingface folder, there is one for each node



**Using HF models require installation of additional packages:**
- transformers: HF library for transformer models for various tasks


## Parameters to specify model (from Hugging Face Documentation)
**task** (str) — The task defining which pipeline will be returned. Currently accepted tasks are:
- audio-classification": will return a AudioClassificationPipeline.
- "automatic-speech-recognition": will return a AutomaticSpeechRecognitionPipeline.
- "conversational": will return a ConversationalPipeline.
- "depth-estimation": will return a DepthEstimationPipeline.
- "document-question-answering": will return a DocumentQuestionAnsweringPipeline.
- "feature-extraction": will return a FeatureExtractionPipeline.
- "fill-mask": will return a FillMaskPipeline:.
- "image-classification": will return a ImageClassificationPipeline.
- "image-segmentation": will return a ImageSegmentationPipeline.
- "image-to-text": will return a ImageToTextPipeline.
- "mask-generation": will return a MaskGenerationPipeline.
- "object-detection": will return a ObjectDetectionPipeline.
- "question-answering": will return a QuestionAnsweringPipeline.
- "summarization": will return a SummarizationPipeline.
- "table-question-answering": will return a TableQuestionAnsweringPipeline.
- "text2text-generation": will return a Text2TextGenerationPipeline.
" text-classification" (alias "sentiment-analysis" available): will return a TextClassificationPipeline.
- "text-generation": will return a TextGenerationPipeline:.
- "text-to-audio" (alias "text-to-speech" available): will return a TextToAudioPipeline:.
- "token-classification" (alias "ner" available): will return a TokenClassificationPipeline.
- "translation": will return a TranslationPipeline.
- "translation_xx_to_yy": will return a TranslationPipeline.
- "video-classification": will return a VideoClassificationPipeline.
- "visual-question-answering": will return a VisualQuestionAnsweringPipeline.
- "zero-shot-classification": will return a ZeroShotClassificationPipeline.
- "zero-shot-image-classification": will return a ZeroShotImageClassificationPipeline.
- "zero-shot-audio-classification": will return a ZeroShotAudioClassificationPipeline.
- "zero-shot-object-detection": will return a ZeroShotObjectDetectionPipeline.

**model** (str or PreTrainedModel or TFPreTrainedModel, optional) — The model that will be used by the pipeline to make predictions. This can be a model identifier or an actual instance of a pretrained model inheriting from PreTrainedModel (for PyTorch) or TFPreTrainedModel (for TensorFlow).

