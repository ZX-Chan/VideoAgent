# 🎬 VideoAgent: AI-Powered Academic Paper to Video Pipeline

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.8+-blue.svg" alt="Python Version">
  <img src="https://img.shields.io/badge/License-MIT-green.svg" alt="License">
  <img src="https://img.shields.io/badge/AI-Multi--Agent-orange.svg" alt="AI Multi-Agent">
  <img src="https://img.shields.io/badge/Framework-CAMEL-red.svg" alt="CAMEL Framework">
</p>

**VideoAgent** is an intelligent multi-agent system that transforms academic papers into engaging video presentations with automated narration, dynamic animations, and professional slide generation.

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
- [🎬 Manim Animation](#manim-animation)
- [📊 Advanced Usage](#advanced-usage)
- [🔧 Troubleshooting](#troubleshooting)
- [🤝 Contributing](#contributing)

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

### System Dependencies

**LibreOffice Installation:**
```bash
# With sudo access
sudo apt install libreoffice

# Without sudo access
# Download from https://www.libreoffice.org/download/
# Add executable to your $PATH
```

**Poppler for PDF Processing:**
```bash
conda install -c conda-forge poppler
```

**Manim for Animations:**
```bash
pip install manim
```

**FFmpeg for Video Processing:**
```bash
sudo apt install ffmpeg
```

## ⚙️ Configuration

### Environment Variables

Create a `.env` file in the project root:

```bash
# OpenAI API (for GPT models and TTS)
OPENAI_API_KEY=your_openai_api_key

# Gemini API (optional)
GEMINI_API_KEY=your_gemini_api_key

# DeepInfra API (optional)
DEEPINFRA_API_KEY=your_deepinfra_api_key

# Alternative OpenAI-compatible APIs
OPENAI_API_BASE_URL=https://api.nuwaapi.com  # optional
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

**High Performance (GPT-4o):**
```bash
python -m core.new_pipeline_multi \
    --poster_path="data/{paper_name}/paper.pdf" \
    --model_name_t="4o" \
    --model_name_v="4o" \
    --generate \
    --fps=1
```

**Economic (Mixed Models):**
```bash
python -m core.new_pipeline_multi \
    --poster_path="data/{paper_name}/paper.pdf" \
    --model_name_t="qwen-2.5-vl-7b" \
    --model_name_v="4o" \
    --generate \
    --fps=1
```

**Local Deployment (Qwen Models):**
```bash
# First, start vLLM service
python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen2.5-VL-7B-Instruct \
    --trust-remote-code \
    --max-model-len 4096 \
    --port 8000

# Then run VideoAgent
python -m core.new_pipeline_multi \
    --poster_path="data/{paper_name}/paper.pdf" \
    --model_name_t="qwen-2.5-vl-7b" \
    --model_name_v="qwen-2.5-vl-7b" \
    --generate \
    --fps=1
```

### 3. Advanced Options

**With Bullet Points:**
```bash
python -m core.new_pipeline_multi \
    --poster_path="data/{paper_name}/paper.pdf" \
    --model_name_t="4o" \
    --model_name_v="4o" \
    --generate \
    --use_bullet_points \
    --fps=1
```

**Skip Image Filtering:**
```bash
python -m core.new_pipeline_multi \
    --poster_path="data/{paper_name}/paper.pdf" \
    --model_name_t="4o" \
    --model_name_v="4o" \
    --generate \
    --filter \
    --fps=1
```

## 🎬 Manim Animation

VideoAgent features AI-powered Manim animation generation using a two-agent system:

### Enable Manim Animations

```bash
python -m core.new_pipeline_multi \
    --poster_path="data/{paper_name}/paper.pdf" \
    --model_name_t="4o" \
    --model_name_v="4o" \
    --generate \
    --use_manim \
    --fps=1
```

### How It Works

1. **Agent1 (Animation Planner)**: Analyzes paper content and creates detailed animation plans
2. **Agent2 (Code Generator)**: Generates executable Manim code based on the plan and figure images
3. **Automatic Integration**: Seamlessly integrates animations into the final video

### Animation Features

- 🎯 **Smart Figure Analysis**: Automatically identifies key figures for animation
- 🎨 **Dynamic Visualizations**: Creates mathematical animations and data flow demonstrations
- 🔧 **Code Validation**: Ensures generated Manim code is syntactically correct
- 📐 **Layout Optimization**: Adapts animations to fit presentation layout

## 📊 Advanced Usage

### Custom Poster Dimensions

```bash
python -m core.new_pipeline_multi \
    --poster_path="data/{paper_name}/paper.pdf" \
    --poster_width_inches=56 \
    --poster_height_inches=42 \
    --model_name_t="4o" \
    --generate
```

### Batch Processing

```bash
# Process multiple papers
for paper in data/*/; do
    python -m core.new_pipeline_multi \
        --poster_path="${paper}paper.pdf" \
        --model_name_t="4o" \
        --generate
done
```

### Resume from Existing Results

```bash
# Skip generation, only create video from existing PPT
python -m core.new_pipeline_multi \
    --poster_path="data/{paper_name}/paper.pdf" \
    --model_name_t="4o" \
    --fps=1
```

### Ablation Studies

```bash
# Disable tree layout algorithm
python -m core.new_pipeline_multi \
    --poster_path="data/{paper_name}/paper.pdf" \
    --ablation_no_tree_layout \
    --generate

# Disable commenter agent
python -m core.new_pipeline_multi \
    --poster_path="data/{paper_name}/paper.pdf" \
    --ablation_no_commenter \
    --generate
```

## 🔧 Troubleshooting

### Common Issues

**1. vLLM Service Connection Error:**
```bash
# Check if vLLM is running
curl http://localhost:8000/v1/models

# Restart vLLM service
kill -9 $(ps aux | grep 'vllm' | awk '{print $2}')
python -m vllm.entrypoints.openai.api_server ...
```

**2. GPU Memory Issues:**
```bash
# Monitor GPU usage
nvidia-smi

# Clear GPU cache
python -c "import torch; torch.cuda.empty_cache()"
```

**3. Manim Rendering Errors:**
```bash
# Check Manim installation
manim --version

# Test Manim with simple scene
manim -ql test_scene.py TestScene
```

### Performance Optimization

- **Memory Management**: Configure GPU memory allocation in `core/parse_raw.py`
- **Parallel Processing**: Use `--max_workers` for batch processing
- **Model Selection**: Choose appropriate model combinations based on your hardware

### Debug Mode

```bash
# Enable detailed logging
export PYTHONPATH=$PYTHONPATH:.
python -m core.new_pipeline_multi \
    --poster_path="data/{paper_name}/paper.pdf" \
    --model_name_t="4o" \
    --generate \
    --verbose
```

## 🤝 Contributing

We welcome contributions to VideoAgent! Here's how you can help:

### Development Setup

```bash
git clone https://github.com/your-repo/VideoAgent.git
cd VideoAgent
pip install -r requirements.txt
pip install -e .
```

### Code Style

- Follow PEP 8 guidelines
- Use type hints where appropriate
- Add docstrings for all functions
- Write unit tests for new features

### Submitting Changes

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

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

---

<p align="center">
  Made with ❤️ for the academic community
</p>
