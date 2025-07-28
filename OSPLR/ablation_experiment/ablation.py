import os
import re
import json
import time
from pathlib import Path
from typing import List
from Ablation.DS_API import call_openai
from Ablation.ablation_prompt import build_prompt
def main():
    SRC_TXT = r"C:\Users\卢航青\PycharmProjects\pythonProject11\SPLR实验数据\15%扰动未处理句子.txt"  # 输入的纯文本文件，每行一个待处理句子
    OUTPUT_JSON = r"C:\Users\卢航青\PycharmProjects\pythonProject11\知识库书籍\模式1\实验2——10/15.jsonl"  # 最终将所有行的提取结果写入此 JSON 文件
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
