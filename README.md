# 🎩 Majordome – Your Local AI Voice Butler

**Majordome** is a Speech-to-Speech (STS) local AI butler designed to empower users with an elegant and convenient voice
interaction experience. Built to run entirely on consumer-grade hardware, Majordome brings powerful AI capabilities
directly to your device without relying on cloud services.

## ✨ Features

- **🎤 Real-time Speech Recognition**: Powered by Faster Whisper for accurate voice-to-text transcription
- **🧠 Local Language Model**: Uses Llama 3.2 1B Instruct for intelligent, context-aware responses
- **🔊 Natural Text-to-Speech**: French voice synthesis via Piper TTS with natural intonation
- **⚡ Optimized Performance**: Runs efficiently on consumer-grade devices (CPU or GPU)
- **🔒 Privacy-First**: All processing happens locally - no data leaves your machine
- **🇫🇷 French Language Support**: Fully optimized for French language interactions
- **🛑 Interruptible Responses**: Stop and restart conversations naturally
- **🎯 Voice Activity Detection**: Smart silence detection to reduce false triggers

## 🏗️ Architecture

Majordome uses a multithreaded pipeline architecture:

1. **Audio Capture Thread**: Continuously captures audio from your microphone
2. **VAD + ASR Thread**: Detects speech and transcribes it using Whisper
3. **LLM + TTS Thread**: Generates responses via Llama and speaks them using Piper

This design ensures low latency and responsive interactions while maximizing resource efficiency.

## 📋 Requirements

- **Python**: 3.10.12
- **OS**: Linux, macOS, or Windows
- **Hardware**:
    - CPU: Multi-core processor recommended
    - RAM: Minimum 8GB (16GB recommended)
    - GPU: Optional (CUDA-compatible for faster processing)
    - Microphone and speakers/headphones

## 🚀 Installation

### 1. Clone the Repository

```bash
git clone git@github.com:alexandre-hellin/majordome.git
cd majordome
```

### 2. Create a Virtual Environment

```bash
python -m venv .venv
```

Activate it:

- **Linux / macOS**: `source .venv/bin/activate`
- **Windows**: `.venv\Scripts\activate`

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

> **GPU acceleration (optional)**: If you have a CUDA-compatible GPU, install the matching `torch` and `llama-cpp-python` builds for your CUDA version before running the above command. See the [PyTorch](https://pytorch.org/get-started/locally/) and [llama-cpp-python](https://github.com/abetlen/llama-cpp-python#installation-with-hardware-acceleration) install guides.

### 4. Download Models

Models are **not included** in this repository and must be downloaded separately. Place all files in the `models/` directory.

#### Whisper (Speech Recognition)

Faster Whisper downloads the `small` model automatically on first run and caches it locally. No manual download is required.

#### Llama 3.2 1B Instruct — GGUF (Language Model)

Download the quantized model file from Hugging Face:

```
https://huggingface.co/lmstudio-community/Llama-3.2-3B-Instruct-GGUF
```

Download the file named **`Llama-3.2-3B-Instruct-Q4_K_M.gguf`** and place it at:

```
models/Llama-3.2-3B-Instruct-Q4_K_M.gguf
```

#### Piper TTS — French Voice (Text-to-Speech)

Use the Piper CLI (available after installing dependencies) to download the French UPMC medium voice directly into `models/`:

```bash
python -m piper.download_voices --download-dir models fr_FR-upmc-medium
```

This will place both required files in `models/`:

```
models/fr_FR-upmc-medium.onnx
models/fr_FR-upmc-medium.onnx.json
```

#### Expected `models/` directory layout

```
models/
├── Llama-3.2-3B-Instruct-Q4_K_M.gguf
├── fr_FR-upmc-medium.onnx
└── fr_FR-upmc-medium.onnx.json
```

### 5. Run Majordome

```bash
python main.py
```

Speak into your microphone after startup. Majordome will transcribe your speech, generate a response, and reply aloud in French.

## ⚠️ Disclaimers

Majordome is not the creator, originator, or owner of any third-party model used with this project. Each model (including but not limited to Llama, Whisper, and Piper voices) is created and provided by independent third parties. Majordome does not endorse, support, represent, or guarantee the completeness, truthfulness, accuracy, or reliability of any such model.

You understand that these models can produce content that might be offensive, harmful, inaccurate, inappropriate, or deceptive. Each model is the sole responsibility of the person or entity who originated it. Majordome does not monitor or control third-party models and cannot take responsibility for their outputs.

Majordome disclaims all warranties or guarantees about the accuracy, reliability, or benefits of any third-party model used with this software. Majordome further disclaims any warranty that such models will meet your requirements, be secure, uninterrupted, available, or error-free. You will be solely responsible for any damage resulting from your use of or access to any model downloaded or used in conjunction with this project.

## 🛠️ Troubleshooting

- **No audio input detected**: Ensure your microphone is set as the default input device in your OS audio settings.
- **Slow responses on CPU**: The LLM runs on CPU by default. GPU acceleration requires a CUDA build of `llama-cpp-python` (see step 3).
- **`models/` file not found errors**: Double-check file names match exactly — they are case-sensitive on Linux/macOS.
