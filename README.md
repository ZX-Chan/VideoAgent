# 🎬 VideoAgent: Personalized Synthesis of Scientific Videos

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.8+-blue.svg" alt="Python Version">
  <img src="https://img.shields.io/badge/License-MIT-green.svg" alt="License">
  <img src="https://img.shields.io/badge/AI-Multi--Agent-orange.svg" alt="AI Multi-Agent">
  <img src="https://img.shields.io/badge/Framework-CAMEL-red.svg" alt="CAMEL Framework">
</p>

**VideoAgent** is an intelligent multi-agent system that transforms academic papers into engaging video presentations with automated narration, dynamic animations, and professional slide generation.
<img width="1831" height="604" alt="image" src="https://github.com/user-attachments/assets/356aa64f-c557-452b-9459-e77b4fafeed7" />

## ✨ Key Features

- 🤖 **Multi-Agent Architecture**: Leverages CAMEL framework with specialized agents for different tasks
- 📄 **Paper Parsing**: Intelligent extraction of content, figures, and tables from PDF papers
- 🎨 **Automated PPT Generation**: Creates professional multi-slide presentations
- 🎙️ **Text-to-Speech**: Generates natural narration using OpenAI TTS
- 🎬 **Manim Integration**: Creates dynamic mathematical animations
- 🎯 **Smart Layout**: AI-driven layout optimization with tree-based algorithms
- 🔧 **Flexible Models**: Supports various LLM/VLM combinations (GPT-4, Qwen, Gemini)

## 📋 Table of Contents

- [🛠️ Installation](#installation)
- [⚙️ Configuration](#configuration)
- [🚀 Quick Start](#quick-start)


---

## 🛠️ Installation

VideoAgent supports both local deployment (via [vLLM](https://docs.vllm.ai/en/v0.6.6/getting_started/installation.html)) and API-based access (GPT-4, Gemini, etc.).

### Prerequisites

- Python 3.8+
- CUDA-compatible GPU (for local models)
- LibreOffice for PPT processing
- FFmpeg for video generation

### Python Dependencies

```bash
git clone https://github.com/your-repo/VideoAgent.git
cd VideoAgent
pip install -r requirements.txt
```



## ⚙️ Configuration

### Environment Variables

Create a `.env` file in the project root:

```bash
# OpenAI API for Instance
OPENAI_API_KEY=your_openai_api_key
```

### Model Configuration

Edit model configurations in `utils/wei_utils.py`:

```python
def get_agent_config(model_name):
    if model_name == "4o":
        return {
            "model_platform": ModelPlatformType.OPENAI,
            "model_type": "gpt-4o",
            # ... other configs
        }
```

---

## 🚀 Quick Start

### 1. Prepare Your Paper

Create a folder structure for your paper:
```
📁 data/
└── 📁 {paper_name}/
    └── 📄 paper.pdf
```

### 2. Basic Video Generation

```bash
python -m core.new_pipeline_multi \
    --poster_path="data/{paper_name}/paper.pdf" \
    --model_name_t="Preferred_Model" \
    --poster_width_inches=48 \
    --poster_height_inches=36 \
    --ablation_no_tree_layout \
    --use_bullet_points \
    --generate \
    --use_manim
```



## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## ❤️ Acknowledgements

We extend our gratitude to:

- [🐫 CAMEL](https://github.com/camel-ai/camel) - Multi-agent framework
- [📄 Docling](https://github.com/docling-project/docling) - Document parsing
- [🎬 Manim](https://github.com/3b1b/manim) - Mathematical animations
- [🎤 OpenAI](https://openai.com/) - Text-to-speech and language models
- [⚡ vLLM](https://github.com/vllm-project/vllm) - High-performance LLM serving

## 📧 Contact

For questions, suggestions, or collaboration opportunities:

- 📧 Email: [your-email@domain.com]
- 🐛 Issues: [GitHub Issues](https://github.com/your-repo/VideoAgent/issues)
- 💬 Discussions: [GitHub Discussions](https://github.com/your-repo/VideoAgent/discussions)

