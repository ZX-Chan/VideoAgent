# 🎬 VideoAgent: Síntesis Personalizada de Videos Científicos

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.12-blue.svg" alt="Python Version">
  <img src="https://img.shields.io/badge/License-MIT-green.svg" alt="License">
  <img src="https://img.shields.io/badge/AI-Multi--Agent-orange.svg" alt="AI Multi-Agent">
  <img src="https://img.shields.io/badge/Framework-CAMEL-red.svg" alt="CAMEL Framework">
</p>

**VideoAgent** es un sistema inteligente de múltiples agentes que transforma artículos académicos en presentaciones de video atractivas con narración automatizada, animaciones dinámicas y generación profesional de diapositivas.
<img width="1831" height="604" alt="image" src="https://github.com/user-attachments/assets/356aa64f-c557-452b-9459-e77b4fafeed7" />

## ✨ Características Principales

- 🤖 **Arquitectura Multi-Agente**: Aprovecha el framework CAMEL con agentes especializados para diferentes tareas
- 📄 **Análisis de Artículos**: Extracción inteligente de contenido, figuras y tablas de artículos en PDF
- 🎨 **Generación Automática de PPT**: Crea presentaciones profesionales de múltiples diapositivas
- 🎙️ **Texto a Voz**: Genera narración natural utilizando OpenAI TTS
- 🎬 **Integración con Manim**: Crea animaciones matemáticas dinámicas
- 🎯 **Diseño Inteligente**: Optimización de diseño impulsada por IA con algoritmos basados en árboles
- 🔧 **Modelos Flexibles**: Admite diversas combinaciones de LLM/VLM (GPT-4, Qwen, Gemini)

## 📋 Tabla de Contenidos

- [🛠️ Instalación](#installation)
- [⚙️ Configuración](#configuration)
- [🚀 Inicio Rápido](#quick-start)
- [📄 Conjunto de Datos SciVidEval](#dataset)


---
## 📜 Conjunto de Datos SciVidEval

Este conjunto de datos está diseñado para evaluación.

**Evaluación**: Los scripts y herramientas de evaluación se pueden encontrar en el directorio [VideoAgent/SciVidEval](https://github.com/ZX-Chan/VideoAgent/tree/main/SciVidEval).

**Descarga de Datos**:
* **BaiduYun**: [Enlace](https://pan.baidu.com/s/1nnR2HbWSsNMAR2_i6ZOssQ?pwd=kezv)
* **Google Drive**: [Enlace](https://drive.google.com/file/d/1J8pUAOWlOo55fklxQ3VkcJukIQSBPjJi/view?usp=drive_link)

**Estructura de Carpetas**:
```
SciVidEval/
├── videos/
│   └── ACL2022_01.mp4 
│   └── ...
├── papers/
│   └── ACL2022_01.pdf
│   └── ...
├── processed/
│   └── ACL2022_01/
│       ├── audio/
│       ├── frame/
│       ├── ACL2022_01_gpt.json
│       ├── pipeline_log.txt
│       └── state.json
└── (otros pares de video/artículo)

```




## 🛠️ Instalación

VideoAgent admite tanto el despliegue local (a través de [vLLM](https://docs.vllm.ai/en/v0.6.6/getting_started/installation.html)) como el acceso basado en API (GPT-4, Gemini, etc.).

### Requisitos Previos

- Python 3.12
- GPU compatible con CUDA (para modelos locales)
- LibreOffice para el procesamiento de PPT
- FFmpeg para la generación de video

### Dependencias de Python

```bash
cd VideoAgent
pip install -r requirements.txt
```



## ⚙️ Configuración

### Variables de Entorno

Crea un archivo `.env` en la raíz del proyecto:

```bash
# OpenAI API para Instancia
OPENAI_API_KEY=your_openai_api_key
```

### Configuración del Modelo

Edita las configuraciones del modelo en `utils/wei_utils.py`:

```python
def get_agent_config(model_name):
    if model_name == "4o":
        return {
            "model_platform": ModelPlatformType.OPENAI,
            "model_type": "gpt-4o",
            # ... otras configuraciones
        }
```

---

## 🚀 Inicio Rápido

### 1. Prepara Tu Artículo

Crea una estructura de carpetas para tu artículo:
```
📁 data/
└── 📁 {nombre_del_artículo}/
    └── 📄 paper.pdf
```

### 2. Generación Básica de Video

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



## 📜 Licencia

Este proyecto está licenciado bajo la Licencia MIT - consulta el archivo [LICENSE](LICENSE) para más detalles.

## ❤️ Reconocimientos

Extendemos nuestro agradecimiento a:

- [🐫 CAMEL](https://github.com/camel-ai/camel) - Framework multi-agente
- [📄 Docling](https://github.com/docling-project/docling) - Análisis de documentos
- [🎬 Manim](https://github.com/3b1b/manim) - Animaciones matemáticas
- [🎤 OpenAI](https://openai.com/) - Texto a voz y modelos de lenguaje
- [⚡ vLLM](https://github.com/vllm-project/vllm) - Servicio de LLM de alto rendimiento
- [📄 Paper2Poster](https://github.com/Paper2Poster/Paper2Poster) - Nuestro trabajo se basa en este increíble proyecto.
