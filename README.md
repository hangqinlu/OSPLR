# OSPLR: A Framework for Complex Entity Recognition and Automated Knowledge Graph Construction from Historical Chinese Texts
## Project Overview

OSPLR addresses persistent challenges in constructing knowledge graphs from historical Chinese texts, particularly the modeling of complex, polysemous, and nested entities, as well as deep attribute and relationship extraction. Traditional methods rely heavily on manual annotation and lack the scalability and accuracy required for large-scale, automatic processing. While Large Language Models (LLMs) show promise for knowledge extraction and reasoning, their effectiveness is fundamentally constrained by the availability and quality of domain-specific pre-training corpora.

To address these challenges, OSPLR introduces a **Single-Point and Length-Representation-based Nested Named Entity Recognition (SPLR)** model, tailored for precise recognition of polysemous and nested entities in historical texts. By incorporating **Chain-of-Thought (CoT) prompting** and explicit ontology-driven priors, OSPLR injects structured entity information into the LLM workflow, enabling fine-grained, ontology-driven reasoning and information extraction.

**The OSPLR pipeline consists of three main stages:**
1. **Structure Constraints:** Ontology-based rules and normalization.
2. **External Knowledge Extraction:** Automatic NER and attribute inference using SPLR.
3. **Reasoning Fusion:** LLM reasoning with chain-of-thought prompt engineering and structured knowledge integration.

---


## Features

- Automatic recognition and extraction of polysemous, nested, and ambiguous entities from historical texts
- Comprehensive ontology modeling for complex attributes and relations
- Ontology-driven Chain-of-Thought (CoT) prompting for enhanced LLM inference and deep reasoning
- End-to-end automation: from data preprocessing to entity recognition, LLM reasoning, and knowledge graph construction
- Modular and extensible architecture: supports easy adaptation, transfer learning, and further development for other domains or languages

---

## Directory Structure
```text
OSPLR/
│
├── SPLR/                     # Core Python package
│   ├── __init__.py
│   ├── ds.py                 # LLM (DeepSeek, OpenAI etc.) API interface
│   ├── model.py              # SPLR model definition
│   ├── inference.py          # Inference pipeline for SPLR/OSPLR
│   ├── osplr_prompt.py       # Prompt generation and formatting
│   └── utils.py              # Utility functions
│
├── experimental_results/     # All experiment results
│   ├── ablation/
│   │   ├── ablation_group1.json
│   │   ├── ablation_group2.json
│   ├── osplr_pred/
│   │   ├── osplr_group1.json
│   │   ├── osplr_group1_5p_rp.json
│   │   ├── osplr_group1_10p_rp.json
│   │   ├── osplr_group1_15p_rp.json
│   │   ├── osplr_group2.json
│   │   ├── osplr_group2_5p_rp.json
│   │   ├── osplr_group2_10p_rp.json
│   │   ├── osplr_group2_15p_rp.json
│   └── calc_method/
│       ├── calculation.py
│       ├── rp_calc.py
│
├── ablation_experiment/      # Ablation experiments and scripts
│   ├── Ablation/
│   │   ├── ablation_prompt.py
│   │   ├── ablation_ds.py
│   └── ablation.py
│
├── data/                     # Datasets
│   ├── gold_data/
│   │   ├── group1.json
│   │   ├── group2.json
│   └── input_text/
│       ├── group1.txt
│       ├── group2.txt
│
├── checkpoints/              # Model checkpoints
│   └── my_full_model.pt
│
├── configs/                  # Configuration files
│   └── config.yaml
│
├── main.py                   # Main pipeline entry
├── README.md                 # This file
├── .gitignore
└── LICENSE

```



---
## Configuration

All file paths and major hyperparameters are set in `config.yaml`:

```yaml
# config.yaml
device: "cuda"

model:
  pretrained_dir: "SIKU-BERT/sikubert"
  checkpoint_path: "checkpoints/my_full_model.pt"
  ner_type_file: "data/SPLRtext"

data:
  input_txt: "data/input_text/group1.txt"
  output_json: "Experimental Results/osplr_pred/osplr_group1.json"

llm:
  api_base_url: "https://tbnx.plus7.plus/v1"
  api_key: "sk-e8DdamFXsM6jBn1MA5NTyUAvMDdsQLJnKLKfgItEz75GUj1Q"         #
  model_name: "deepseek-reasoner"
  timeout: 1500

other:
  sleep_per_sample: 1
  max_retry: 3

```


## Training & Inference

### Training: 
 See SPLR/model.py for model training details.

###  Inference:
 Run main.py for end-to-end knowledge extraction and graph construction.
```bash
python main.py

```
## Citation
``` bibtex
@misc{OSPLR_2025,
  author = {Hangqin Lu},
  title = { OSPLR: A Framework for Complex Entity Recognition and Automated Knowledge Graph Construction from Historical Chinese Texts },
  year = {2025},
  howpublished = {\url{https://github.com/HangqinLu/OSPLR}}
}
```
## License

``` markdown
MIT License, see LICENSE for details.

Copyright (c) 2025 Hangqin Lu

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

---







