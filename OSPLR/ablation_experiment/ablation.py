import os
import re
import json
import time
from pathlib import Path
from typing import List
from Ablation.DS_API import call_openai
from Ablation.ablation_prompt import build_prompt
def main():
    SRC_TXT = r"data\group1.txt"  
    OUTPUT_JSON = "Experimental Results1\ablation_pred/ablation_group1.json"  
    os.makedirs(Path(OUTPUT_JSON).parent, exist_ok=True)
    all_results: List[List[dict]] = []
    with open(SRC_TXT, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f, 1):
            text = line.strip()
            if not text:
                print(f"[WARN] 第 {idx} 行为空行，跳过。")
                continue
            prompot = build_prompt(text)

            print(f"[INFO] ({idx}) 调用 API 处理中 …")
            extracted = call_openai(prompot)
            print(extracted)
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
