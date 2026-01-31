# Ad Embeddings Image Pipeline

Local, modular pipeline for image ad variants with SAM3 segmentation, deterministic overlay generation, optional Qwen editing, acceptability scoring, and Streamlit review UI.

## Requirements
- Python 3.9+
- CPU is fine for non-SAM parts.
- SAM3 may require GPU; the code will warn if CUDA is absent.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### SAM3 (official repo)
Clone the official SAM3 repository and install it in editable mode. Example:

```bash
# from a parent directory
# git clone <official SAM3 repo>
# cd sam3
pip install -e .
```

The pipeline expects the official API:

```python
from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor
model = build_sam3_image_model()
processor = Sam3Processor(model)
```

Set your checkpoint path if required by the repo:

```bash
export SAM3_CHECKPOINT=/path/to/sam3_checkpoint.pth
```

If SAM3 is not installed or the checkpoint is missing, segmentation will skip and instruct you.

### SAM3 via Transformers (optional, HF Space style)
You can also use the Hugging Face Transformers SAM3 API (as used in the `akhaliq/sam3` space).
Install transformers (main branch is sometimes required for SAM3):

```bash
pip install -U git+https://github.com/huggingface/transformers
```

The pipeline will automatically use the Transformers SAM3 backend if it is available.

### Hugging Face Qwen Image Edit (optional)
Set your HF token to enable Qwen editing:

```bash
export HF_TOKEN=hf_your_token
```

If HF_TOKEN is not set, Qwen is disabled and the pipeline uses deterministic overlay.

## Run

```bash
streamlit run app.py
```

Smoke test (1 image):

```bash
python scripts/smoke_test.py
```

## Data layout

```
ad_pipeline/
  data/
    images/raw/        # input images
    images/masks/      # SAM3 masks
    images/variants/   # generated variants + scores
    reviews/           # review state
    exports/           # approved CSV exports
```

## Notes
- Acceptability uses OCR if `pytesseract` is installed. Otherwise it falls back to histogram similarity vs brand reference or a basic image-statistics score.
- Qwen mask conditioning is attempted if the HF endpoint supports it. If not, the code edits the whole image and records that limitation.
