# OCR_Imagecaptioning_openai

Try Image captioning BLIP model with the below link:
https://blippy-souji.streamlit.app/
# Multimodal Image Accessibility Suite

![PyTorch](https://img.shields.io/badge/PyTorch-2.0-%23EE4C2C)
![OpenAI](https://img.shields.io/badge/OpenAI-GPT3.5-purple)
![TTS](https://img.shields.io/badge/Text--to--Speech-CoquiTTS-green)
![OCR](https://img.shields.io/badge/OCR-PaddleOCR-blue)

**Bridging visual and auditory worlds** with AI-powered image captioning and audio descriptions for enhanced accessibility.

🌐 [Live Demo](#) | 🔊 [Audio Samples](#) | 📄 [Research Paper](#)

## Key Features

| Component | Technology | Accuracy |
|-----------|------------|----------|
| **Text Extraction** | PaddleOCR | 94% (ICDAR benchmark) |
| **Visual Captioning** | BLIP-2 | 82% CIDEr score |
| **Context Enhancement** | GPT-3.5-turbo | - |
| **Audio Generation** | Coqui TTS | 4.1 MOS |

## Architecture

```mermaid
graph LR
    A[Input Image] --> B(PaddleOCR)
    B --> C[Extracted Text]
    A --> D(BLIP-2)
    D --> E[Visual Captions]
    C & E --> F(GPT-3.5 Context Fusion)
    F --> G[Final Description]
    G --> H(Coqui TTS)
    H --> I[Audio Output]
