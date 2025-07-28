import json
import time
from pathlib import Path
import os
import torch
from transformers import AutoTokenizer, AutoModel
from SPLR.model import RModel
from SPLR.inference import ner_inference
from SPLR.ds import call_openai
from SPLR.utils import build_type_index
from SPLR.osplr_prompt import build_prompt
import yaml


def load_config():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(base_dir, 'configs', 'config.yaml')
    with open(config_path, "r", encoding="utf-8") as f:
        import yaml
        config = yaml.safe_load(f)
    return config

def main():
    config = load_config()
    model_path = config['model']['checkpoint_path']
    model_dir = config['model']['pretrained_dir']
    ner_type_file = config['model']['ner_type_file']
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    type_2_index, index_2_type = build_type_index(ner_type_file)
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    BERT = AutoModel.from_pretrained(model_dir)
    model = RModel(BERT, len(type_2_index)).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    SRC_TXT = config['data']['input_txt']
    OUTPUT_JSON = config['data']['output_json']
    os.makedirs(Path(OUTPUT_JSON).parent, exist_ok=True)
    all_results = []
    with (open(SRC_TXT, "r", encoding="utf-8") as f):
        for idx, line in enumerate(f, 1):
            text = line.strip()
            if not text:
                print(f"[WARN] 第 {idx} 行为空行，跳过。")
                all_results.append([])
                continue
            print(f"[INFO] ({idx}) NER推理中 ...")
            ner_knowledge = ner_inference(text,tokenizer,device,model,index_2_type)
            print(f"NER外部知识{ner_knowledge}")
            PROMPT_TEMPLATE =build_prompt(text,ner_knowledge)
            print(f"[INFO] ({idx}) LLM推理 ...")
            extracted = call_openai(PROMPT_TEMPLATE)
            if extracted:
                all_results.append(extracted)
                print(f"[OK] 第 {idx} 行结果已获取，共 {len(extracted)} 条 JSON 对象。")
            else:
                all_results.append([])
                print(f"[FAIL] 第 {idx} 行未获得有效输出。")

            time.sleep(1)
    with open(OUTPUT_JSON, "w", encoding="utf-8") as outf:
        json.dump(all_results, outf, ensure_ascii=False, indent=2)

    print(f"=== 所有行处理完成，结果已保存到 {OUTPUT_JSON} ===")

if __name__ == "__main__":
    main()
