# VisionLM
Vision Language model

## VisionLM: A Language Model for visual understanding
VisionLM is a project that combines the OPT language model12 and the VIT clip model34 to create a powerful and versatile model for vision and language tasks. VisionLM is inspired by the LLaVA model, which also leverages pre-trained language and vision models to perform zero-shot image classification, image captioning, visual question answering, and more.

## Features
- VisionLM uses the OPT-1.3B language model, which has 1.3 billion parameters. OPT-1.3b is trained on publicly available datasets and can be used for zero-shot and few-shot learning on various natural language tasks.
- VisionLM uses the VIT-B/32 clip model, which is trained on a large-scale dataset of (image, text) pairs4. VIT-B/32 uses a Transformer architecture to encode images into latent vectors that can be compared with text embeddings via a dot product similarity score.
- VisionLM combines the OPT and VIT models by projecting their embeddings into a shared latent space of 512 dimensions. This allows VisionLM to leverage the complementary strengths of both models and perform cross-modal tasks such as image-text retrieval, image captioning, visual question answering, etc.
- VisionLM can be fine-tuned on specific tasks or domains using gradient-based methods or prompt engineering techniques. VisionLM can also be used in a zero-shot manner by providing natural language instructions as input.

## Usage
To use train the model follow the instructions.
For example:
```
git clone https://github.com/PrAsAnNaRePo/VisionLM
cd VisionLM
python train.py --llm_id "facebook/opt-1.3b" --epochs 2
```
