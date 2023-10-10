# Integrating YOLOv8
## Nodes
- **hf_node: HFNode** -- This node accepts multiple video node and applies specified Hugging Face models on frames of those videos. Need to specify specific model in the configuration file. Currently testing with models with computer vision tasks.
- **hf_display: HFDisplay** -- This display the results alongside the source video.


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

