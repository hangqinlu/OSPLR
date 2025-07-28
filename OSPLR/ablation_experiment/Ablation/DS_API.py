from openai import OpenAI
import re
import json
def is_valid_jsonl(line):
    try:
        json.loads(line)
        return True
    except Exception:
        return False
def clean_stream_response(raw: str) -> str:
    no_think = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
    return re.sub(r"^[^{]*", "", no_think, count=1).strip()

def extract_json_objs(text):
    try:
        data = json.loads(text)
        if isinstance(data, list):
            return data
        elif isinstance(data, dict):
            return [data]
    except Exception:
        pass

    lines = [ln for ln in text.splitlines() if is_valid_jsonl(ln)]
    if len(lines) > 0 and len(lines) >= text.count('{') // 2:
        return [json.loads(ln) for ln in lines]

    pattern = r'(\{[\s\S]*?\})(?=(?:\s*\{)|\s*$)'
    matches = re.finditer(pattern, text)
    objs = []
    for m in matches:
        try:
            obj = json.loads(m.group(1))
            objs.append(obj)
        except Exception:
            continue
    return objs

client = OpenAI(
    base_url="https://tbnx.plus7.plus/v1",
    api_key="sk-e8DdamFXsM6jBn1MA5NTyUAvMDdsQLJnKLKfgItEz75GUj1Q"
)

def call_openai(full_prompt: str) -> list:
    full_response = ""
    try:
        stream = client.chat.completions.create(
            model="deepseek-reasoner",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "你是一个帮助生成明朝科举-人物-职官-文献-地点知识图谱的助手，"
                        "需要从古籍文本中提取结构化数据并生成图数据库格式，"
                        "输出只能包含严格的 JSONL 数据，不能有任何额外信息。注意按行输出，一行为一条数据"
                    )
                },
                {
                    "role": "user",
                    "content": full_prompt
                }
            ],
            stream=True,
            timeout=1500
        )
        for chunk in stream:
            if hasattr(chunk, "choices"):
                for choice in chunk.choices:
                    delta = getattr(choice, "delta", None)
                    if delta and getattr(delta, "content", None):
                        content = delta.content
                        if content.strip():
                            full_response += content
    except Exception as e:
        print(f"[ERROR] OpenAI 调用失败: {e}")
        return []
    cleaned = clean_stream_response(full_response)
    return extract_json_objs(cleaned)