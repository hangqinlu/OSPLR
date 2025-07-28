import json
from collections import defaultdict
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

PRIMARY_KEY = {
    "Person": "姓名",
    "Place": "歷史名稱",
    "OfficialPosition": "官職名稱",
    "ImperialExam": "考試等級"
}
IMPLICIT_ATTRS = {
    "區劃層級", "官階", "現代名稱", "經緯座標",
    "現代區劃層級", "授予類型", "遷轉", "關係類型", "職事名目"
}
SPECIAL_WORDS = set("府縣衛司道直隸")

def strip_special_words(s):
    if not isinstance(s, str):
        s = str(s)
    return ''.join(c for c in s if c not in SPECIAL_WORDS)

def levenshtein(a, b):
    n, m = len(a), len(b)
    if n > m:
        a, b = b, a
        n, m = m, n
    current = list(range(n + 1))
    for i in range(1, m + 1):
        previous, current = current, [i] + [0] * n
        for j in range(1, n + 1):
            add, delete, change = previous[j] + 1, current[j - 1] + 1, previous[j - 1]
            if a[j - 1] != b[i - 1]:
                change += 1
            current[j] = min(add, delete, change)
    return current[n]

def text_similarity(s1, s2):
    if not s1 or not s2:
        return 0.0
    lev = levenshtein(s1, s2)
    max_len = max(len(s1), len(s2))
    sim = 1 - lev / max_len if max_len > 0 else 0.0
    return sim

def find_best_match(text, gold_data, threshold=0.8):
    best_item = None
    best_sim = 0.0
    for g in gold_data:
        gold_text = g.get("Text", "")
        sim = text_similarity(text, gold_text)
        if sim > best_sim:
            best_sim = sim
            best_item = g
    if best_sim >= threshold:
        return best_item
    else:
        return None

def n_char_off(s1, s2, n=1):
    s1 = strip_special_words(s1)
    s2 = strip_special_words(s2)
    if s1 == s2:
        return True
    l1, l2 = len(s1), len(s2)
    if min(l1, l2) <= n:
        return False
    if abs(l1 - l2) <= n:
        if levenshtein(s1, s2) <= n:
            return True
    if l1 == l2 and l1 > n:
        diff = sum(1 for a, b in zip(s1, s2) if a != b)
        if diff <= n:
            return True
    return False

def fuzzy_equal(g, p, field=None, node_label=None):
    if node_label == "Place" and field == "歷史名稱":
        g1 = strip_special_words(g)
        p1 = strip_special_words(p)
        return g1 == p1
    return n_char_off(g, p, n=1)

def relation_key_contains(g, p):
    def get_start_end(key):
        _, rest = key.split('-', 1)
        start, end = rest.split('-')
        start_val = start.split(':', 1)[-1]
        end_val = end.split(':', 1)[-1]
        start_val = strip_special_words(start_val)
        end_val = strip_special_words(end_val)
        return start, start_val, end, end_val

    g_start, g_start_val, g_end, g_end_val = get_start_end(g)
    p_start, p_start_val, p_end, p_end_val = get_start_end(p)
    if g.startswith("社會關係") and p.startswith("社會關係"):
        direct = ((g_start_val in p_start_val or p_start_val in g_start_val) and
                  (g_end_val in p_end_val or p_end_val in g_end_val))
        swap = ((g_start_val in p_end_val or p_end_val in g_start_val) and
                (g_end_val in p_start_val or p_end_val in g_end_val))
        return direct or swap
    else:
        start_match = (g_start_val in p_start_val) or (p_start_val in g_start_val)
        end_match = (g_end_val in p_end_val) or (p_end_val in g_end_val)
        return start_match and end_match

def extract_nodes_by_label(nodes):
    res = defaultdict(list)
    for node in nodes:
        label = node['label']
        props = node['properties']
        key = PRIMARY_KEY.get(label)
        v = props.get(key)
        if label == "Place" and (v == "雲南" or v == "云南"):
            continue
        if key and v not in [None, "", "null", "None"]:
            res[label].append(str(v))
    return res

def extract_attrs_by_name(nodes):
    attr_map = defaultdict(list)
    for node in nodes:
        label = node['label']
        props = node['properties']
        key_val = props.get(PRIMARY_KEY.get(label, ''), None)
        for k, v in props.items():
            if v in [None, "", "null", "None"] or k in IMPLICIT_ATTRS:
                continue
            if key_val not in [None, "", "null", "None"]:
                attr_map[k].append(f"{label}-{strip_special_words(key_val)}-{k}:{strip_special_words(v) if k=='歷史名稱' else v}")
    return attr_map

def extract_relation_properties_flat(relationships):
    rel_attr_map = defaultdict(list)
    for rel in relationships:
        label = rel['label']
        start = rel['start']
        end = rel['end']
        if start['label'] in PRIMARY_KEY and end['label'] in PRIMARY_KEY:
            start_key = start.get(PRIMARY_KEY[start['label']])
            end_key = end.get(PRIMARY_KEY[end['label']])
            triple_prefix = f"{label}-{start['label']}:{strip_special_words(start_key)}-{end['label']}:{strip_special_words(end_key)}"
            prop = {k: v for k, v in rel.get('properties', {}).items() if k not in IMPLICIT_ATTRS}
            for k, v in prop.items():
                if v in [None, "", "null", "None"]:
                    continue
                rel_attr_map[k].append(f"{triple_prefix}:{k}:{strip_special_words(v) if k=='歷史名稱' else v}")
    return rel_attr_map

def extract_relations_by_label(relationships):
    rel_map = defaultdict(list)
    for rel in relationships:
        label = rel['label']
        start = rel['start']
        end = rel['end']
        if start['label'] in PRIMARY_KEY and end['label'] in PRIMARY_KEY:
            start_key = start.get(PRIMARY_KEY[start['label']])
            end_key = end.get(PRIMARY_KEY[end['label']])
            rel_map[label].append(f"{label}-{start['label']}:{strip_special_words(start_key)}-{end['label']}:{strip_special_words(end_key)}")
    return rel_map

def match_and_count(gold, pred, equal_func, context=None):
    tp = 0
    gold_used = [False] * len(gold)
    pred_used = [False] * len(pred)
    fn_list = []
    fp_list = []
    tp_pair = []
    for i, g in enumerate(gold):
        matched = False
        for j, p in enumerate(pred):
            if pred_used[j]:
                continue
            if equal_func(g, p, **(context or {})):
                tp += 1
                gold_used[i] = True
                pred_used[j] = True
                tp_pair.append((g, p))
                matched = True
                break
        if not matched:
            fn_list.append(g)
    for j, used in enumerate(pred_used):
        if not used:
            fp_list.append(pred[j])
    return tp, fp_list, fn_list, tp_pair

def prf(tp, fp, fn):
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    return prec, rec, f1

def struct_f1_single(gold_nodes, gold_attrs, gold_rels, pred_nodes, pred_attrs, pred_rels):
    def flat_node(nodes):
        for label, vals in nodes.items():
            for v in vals:
                yield f"node-{label}:{strip_special_words(v)}"
    def flat_attr(attrs):
        for attr, vals in attrs.items():
            for v in vals:
                yield f"attr-{attr}:{v}"
    def flat_rel(rels):
        for rel_label, vals in rels.items():
            for v in vals:
                yield f"relkey-{rel_label}:{v}"
    gold_struct = set(list(flat_node(gold_nodes)) + list(flat_attr(gold_attrs)) + list(flat_rel(gold_rels)))
    pred_struct = set(list(flat_node(pred_nodes)) + list(flat_attr(pred_attrs)) + list(flat_rel(pred_rels)))
    tp = len(gold_struct & pred_struct)
    fp = len(pred_struct - gold_struct)
    fn = len(gold_struct - pred_struct)
    return prf(tp, fp, fn)

if __name__ == "__main__":
    gold_path = r'C:\Users\卢航青\PycharmProjects\pythonProject11\SPLR实验数据\科举——组别2.json'
    pred_path = r"C:\Users\卢航青\PycharmProjects\pythonProject11\SPLR实验数据\SPLR_繁体——段落临时扰动15.json"

    with open(gold_path, 'r', encoding='utf-8') as f:
        gold = json.load(f)
    with open(pred_path, 'r', encoding='utf-8') as f:
        pred = json.load(f)

    node_detail = defaultdict(lambda: {'tp':0, 'fp':0, 'fn':0})
    attr_detail = defaultdict(lambda: {'tp':0, 'fp':0, 'fn':0})
    relkey_detail = defaultdict(lambda: {'tp':0, 'fp':0, 'fn':0})

    all_ex_attr_gold = []
    all_ex_attr_pred = []
    all_gold_total = set()
    all_pred_total = set()

    struct_f1s = []
    node_f1s = []
    attr_f1s = []
    rel_f1s = []

    print("="*40)
    print("每条数据性能统计：")
    print("="*40)
    for idx, pred_item in enumerate(pred):
        pred_obj = pred_item[0] if isinstance(pred_item, list) and pred_item else pred_item if isinstance(pred_item, dict) else None
        if not pred_obj:
            struct_f1s.append(0)
            node_f1s.append(0)
            attr_f1s.append(0)
            rel_f1s.append(0)
            continue
        text = pred_obj.get("Text", None)
        if text is None:
            struct_f1s.append(0)
            node_f1s.append(0)
            attr_f1s.append(0)
            rel_f1s.append(0)
            continue
        # --------- 修改：用模糊匹配Text对齐 ----------
        gold_item = find_best_match(text, gold, threshold=0.8)
        # ---------------------------------------------
        pred_nodes = extract_nodes_by_label(pred_obj.get('nodes', []))
        gold_nodes = extract_nodes_by_label(gold_item.get('nodes', [])) if gold_item else defaultdict(list)
        pred_attrs = extract_attrs_by_name(pred_obj.get('nodes', []))
        gold_attrs = extract_attrs_by_name(gold_item.get('nodes', [])) if gold_item else defaultdict(list)
        pred_rels = extract_relations_by_label(pred_obj.get('relationships', []))
        gold_rels = extract_relations_by_label(gold_item.get('relationships', [])) if gold_item else defaultdict(list)
        pred_rel_attr = extract_relation_properties_flat(pred_obj.get('relationships', []))
        gold_rel_attr = extract_relation_properties_flat(gold_item.get('relationships', [])) if gold_item else defaultdict(list)

        # 节点性能
        tp_node = fp_node = fn_node = 0
        for label in set(gold_nodes.keys()) | set(pred_nodes.keys()):
            tp, fp, fn, tp_pair = match_and_count(
                gold_nodes[label], pred_nodes[label],
                lambda g, p, **kwargs: fuzzy_equal(g, p, node_label=label, field=PRIMARY_KEY.get(label)))
            node_detail[label]['tp'] += tp
            node_detail[label]['fp'] += len(fp)
            node_detail[label]['fn'] += len(fn)
            tp_node += tp
            fp_node += len(fp)
            fn_node += len(fn)
            all_gold_total.update((f"node-{label}:{strip_special_words(v)}" for v in gold_nodes[label]))
            all_pred_total.update((f"node-{label}:{strip_special_words(v)}" for v in pred_nodes[label]))
        node_p, node_r, node_f = prf(tp_node, fp_node, fn_node)
        node_f1s.append(node_f)

        # 显性属性性能
        tp_attr = fp_attr = fn_attr = 0
        for attr in set(gold_attrs.keys()) | set(pred_attrs.keys()):
            tp, fp, fn, tp_pair = match_and_count(
                gold_attrs[attr], pred_attrs[attr],
                lambda g, p, **kwargs: fuzzy_equal(g.split(":", 1)[-1], p.split(":", 1)[-1], field=attr))
            attr_detail[attr]['tp'] += tp
            attr_detail[attr]['fp'] += len(fp)
            attr_detail[attr]['fn'] += len(fn)
            tp_attr += tp
            fp_attr += len(fp)
            fn_attr += len(fn)
            all_ex_attr_gold += gold_attrs[attr]
            all_ex_attr_pred += pred_attrs[attr]
            all_gold_total.update((f"attr-{attr}:{v}" for v in gold_attrs[attr]))
            all_pred_total.update((f"attr-{attr}:{v}" for v in pred_attrs[attr]))
        attr_p, attr_r, attr_f = prf(tp_attr, fp_attr, fn_attr)
        attr_f1s.append(attr_f)

        # 对象属性性能
        tp_rel = fp_rel = fn_rel = 0
        for rel_label in set(gold_rels.keys()) | set(pred_rels.keys()):
            tp, fp, fn, tp_pair = match_and_count(
                gold_rels[rel_label], pred_rels[rel_label],
                lambda g, p, **kwargs: relation_key_contains(g, p))
            relkey_detail[rel_label]['tp'] += tp
            relkey_detail[rel_label]['fp'] += len(fp)
            relkey_detail[rel_label]['fn'] += len(fn)
            tp_rel += tp
            fp_rel += len(fp)
            fn_rel += len(fn)
            all_gold_total.update((f"relkey-{rel_label}:{v}" for v in gold_rels[rel_label]))
            all_pred_total.update((f"relkey-{rel_label}:{v}" for v in pred_rels[rel_label]))
        rel_p, rel_r, rel_f = prf(tp_rel, fp_rel, fn_rel)
        rel_f1s.append(rel_f)

        # 关系属性
        for attr in set(gold_rel_attr.keys()) | set(pred_rel_attr.keys()):
            gold_val_list = gold_rel_attr[attr]
            pred_val_list = pred_rel_attr[attr]
            tp, fp, fn, tp_pair = match_and_count(
                gold_val_list, pred_val_list,
                lambda g, p, **kwargs: fuzzy_equal(g.split(":", 1)[-1], p.split(":", 1)[-1], field=attr))
            attr_detail[attr]['tp'] += tp
            attr_detail[attr]['fp'] += len(fp)
            attr_detail[attr]['fn'] += len(fn)
            all_ex_attr_gold += gold_val_list
            all_ex_attr_pred += pred_val_list
            all_gold_total.update((f"relattr-{attr}:{v}" for v in gold_val_list))
            all_pred_total.update((f"relattr-{attr}:{v}" for v in pred_val_list))

        struct_p, struct_r, struct_f = struct_f1_single(
            gold_nodes, gold_attrs, gold_rels,
            pred_nodes, pred_attrs, pred_rels
        )
        struct_f1s.append(struct_f)
        print(f"\n--- 第{idx+1}条 ---")
        print(f"Text: {text}")
        print(f"节点F1: Precision={node_p:.3f} Recall={node_r:.3f} F1={node_f:.3f}")
        print(f"显性属性F1: Precision={attr_p:.3f} Recall={attr_r:.3f} F1={attr_f:.3f}")
        print(f"对象属性F1: Precision={rel_p:.3f} Recall={rel_r:.3f} F1={rel_f:.3f}")
        print(f"本体结构F1: Precision={struct_p:.3f} Recall={struct_r:.3f} F1={struct_f:.3f}")

    print("\n" + "="*40)
    print("全局性能统计：")
    print("="*40)
    total_tp = sum(node_detail[label]['tp'] for label in node_detail)
    total_fp = sum(node_detail[label]['fp'] for label in node_detail)
    total_fn = sum(node_detail[label]['fn'] for label in node_detail)
    p, r, f = prf(total_tp, total_fp, total_fn)
    print(f"总节点F1: Precision={p:.3f} Recall={r:.3f} F1={f:.3f}")
    total_tp = sum(attr_detail[attr]['tp'] for attr in attr_detail)
    total_fp = sum(attr_detail[attr]['fp'] for attr in attr_detail)
    total_fn = sum(attr_detail[attr]['fn'] for attr in attr_detail)
    p, r, f = prf(total_tp, total_fp, total_fn)
    print(f"总显性属性F1: Precision={p:.3f} Recall={r:.3f} F1={f:.3f}")
    total_tp = sum(relkey_detail[rel]['tp'] for rel in relkey_detail)
    total_fp = sum(relkey_detail[rel]['fp'] for rel in relkey_detail)
    total_fn = sum(relkey_detail[rel]['fn'] for rel in relkey_detail)
    p, r, f = prf(total_tp, total_fp, total_fn)
    print(f"总对象属性F1: Precision={p:.3f} Recall={r:.3f} F1={f:.3f}")

    def f1_set(pred_set, gold_set):
        tp = len(pred_set & gold_set)
        fp = len(pred_set - gold_set)
        fn = len(gold_set - pred_set)
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        return prec, rec, f1

    prec, rec, f1 = f1_set(all_pred_total, all_gold_total)
    print(f"整体本体结构F1: Precision={prec:.3f} Recall={rec:.3f} F1={f1:.3f}")

    print("="*40)
    for label in sorted(node_detail.keys()):
        p, r, f = prf(node_detail[label]['tp'], node_detail[label]['fp'], node_detail[label]['fn'])
        print(f"[节点] {label}: Precision={p:.3f} Recall={r:.3f} F1={f:.3f}")
    for attr in sorted(attr_detail.keys()):
        p, r, f = prf(attr_detail[attr]['tp'], attr_detail[attr]['fp'], attr_detail[attr]['fn'])
        print(f"[属性] {attr}: Precision={p:.3f} Recall={r:.3f} F1={f:.3f}")
    for rel_label in sorted(relkey_detail.keys()):
        p, r, f = prf(relkey_detail[rel_label]['tp'], relkey_detail[rel_label]['fp'], relkey_detail[rel_label]['fn'])
        print(f"[对象属性] {rel_label}: Precision={p:.3f} Recall={r:.3f} F1={f:.3f}")

    plt.figure(figsize=(14, 5))
    plt.plot(range(1, len(node_f1s)+1), node_f1s, marker='o', label="节点F1")
    plt.plot(range(1, len(attr_f1s)+1), attr_f1s, marker='s', label="显性属性F1")
    plt.plot(range(1, len(rel_f1s)+1), rel_f1s, marker='^', label="对象属性F1")
    plt.xlabel("数据编号")
    plt.ylabel("F1-score")
    plt.ylim(0, 1)
    plt.title("")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
