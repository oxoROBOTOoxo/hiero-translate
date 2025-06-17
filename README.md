# Hiero-Translate 🚀

Computer-vision + NLP pipeline that takes a photo of Egyptian hieroglyphs and
returns transliteration and English translation.

* **Detector** – YOLOv8/DETR crops individual glyphs  
* **Classifier** – ViT or ConvNeXt assigns Gardiner codes  
* **Translator** – Transformer (LoRA-fine-tuned) maps sequences to text  


## Quick start
```bash
conda env create -f envs/environment.yml
conda activate hiero