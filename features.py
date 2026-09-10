#!/usr/bin/env python3
"""
用 ViennaRNA 从 siRNA 专利数据（v56_*_context10.csv）生成 ASOMAR 所需的特征，
输出与 ASOMAR train.csv 相同 schema 的 CSV。

映射说明见 build_report.md / issue 评论。
"""
import csv
import sys
import re
import RNA

# ---------- 序列工具 ----------
def to_dna(seq: str) -> str:
    return seq.upper().replace('U', 'T')

def revcomp(seq: str) -> str:
    m = {'A': 'U', 'U': 'A', 'G': 'C', 'C': 'G', 'T': 'A'}
    return ''.join(m[c] for c in reversed(seq.upper()))

def gc_frac(seq: str) -> float:
    seq = seq.upper()
    return (seq.count('G') + seq.count('C')) / len(seq) if seq else 0.0

def wallace_tm(seq: str) -> float:
    """Wallace 改进式：Tm = 64.9 + 41*(nGC - 16.4)/N （适用于 ~14-25nt 双链）"""
    s = seq.upper()
    n = len(s)
    ngc = s.count('G') + s.count('C')
    return 64.9 + 41.0 * (ngc - 16.4) / n if n else 0.0

# ---------- 二级结构特征 ----------
def fold_mfe(seq: str):
    """返回 (structure, mfe)"""
    fc = RNA.fold_compound(seq)
    return fc.mfe()

def duplex_energy(a: str, b: str) -> float:
    """两条链间双链自由能 (kcal/mol)"""
    return RNA.duplexfold(a, b).energy

def unpaired_probs(seq: str):
    """配分函数下每个位点未配对(开放)概率，1-based -> 0-based list"""
    fc = RNA.fold_compound(seq)
    fc.pf()                      # 配分函数
    bpp = fc.bpp()               # 基对概率矩阵 (1-indexed)
    n = len(seq)
    p_unpaired = []
    for i in range(1, n + 1):
        paired = sum(bpp[i][j] for j in range(1, n + 1))
        p_unpaired.append(max(0.0, min(1.0, 1.0 - paired)))
    return p_unpaired

def openness_features(seq: str, start: int, length: int):
    """在 seq 的 [start, start+length) 窗口上计算开放度特征（值域 0-1）"""
    if length <= 0 or not seq:
        return 0.0, 0.0, 0.0
    start = max(0, min(start, len(seq) - 1))
    end = min(len(seq), start + length)
    try:
        p = unpaired_probs(seq)
    except Exception:
        return 0.0, 0.0, 0.0
    win = p[start:end]
    if not win:
        return 0.0, 0.0, 0.0
    open_prob = sum(win) / len(win)                       # 平均开放概率
    open_flags = [1 if x > 0.5 else 0 for x in win]
    open_pc = sum(open_flags) / len(win)                  # 开放位点比例
    # 最长连续开放片段占窗口比例
    best = cur = 0
    for f in open_flags:
        cur = cur + 1 if f else 0
        best = max(best, cur)
    max_open_length = best / len(win)
    return open_prob, max_open_length, open_pc

# ---------- 单行转换 ----------
OUT_COLS = ["TS_f50","Sequence","Inhibition","Target_Seq","length","GC","MFE","start.site",
            "TargetRNA","TM","ASOstructure","ASOMFE","dG","concentration","self_bind",
            "TargetGene","modify","Classify","open_prob","max_open_length","open_pc"]

CLASSIFY_THRESHOLD = 0.7   # efficacy >= 0.7 记为高效 (1)

def convert_row(row, idx):
    antisense = row[idx['antisense']].strip().upper()        # RNA, 19-23nt
    target_seq = row[idx['mrna_target_seq']].strip().upper() # DNA 形式靶序列 (21nt)
    context = row[idx['mrna_context_10']].strip().upper() or target_seq
    left_len = int(float(row[idx['mrna_left_valid_length']] or 0))
    target_len = int(float(row[idx['mrna_target_length']] or len(target_seq) or 0))
    efficacy = float(row[idx['efficacy']])

    # 模型输入序列（DNA 化）
    model_seq = to_dna(antisense)

    # 序列自身结构
    aso_struct, aso_mfe = fold_mfe(antisense)
    # 靶局部结构
    _, ctx_mfe = fold_mfe(context)
    # ASO-靶 双链自由能（取绝对值，与原 ASOMAR 正值值域一致）
    try:
        dg = abs(duplex_energy(antisense, target_seq)) if target_seq else 0.0
    except Exception:
        dg = 0.0
    # 自二聚（自结合，正值）
    try:
        self_bind = -duplex_energy(antisense, revcomp(antisense))
    except Exception:
        self_bind = 0.0
    # 靶点开放度
    open_prob, max_open_length, open_pc = openness_features(context, left_len, target_len)

    gene = (row[idx['mrna_target_gene']].strip() or row[idx['target']].strip())
    transcript = row[idx['mrna_transcript']].strip() or row[idx['target']].strip()

    # modify 类别：按 antisense 化学修饰组合（GNA/DNA 存在性）分 4 类，保留修饰信息
    raw_mod = row[idx['antisense_mod_by_position']]
    mod_set = set(re.findall(r'"([A-Za-z0-9]+)"', raw_mod))
    has_gna = 'GNA' in mod_set
    has_dna = 'DNA' in mod_set
    if has_gna and has_dna:
        mod_class = 'GNA+DNA'
    elif has_gna:
        mod_class = 'GNA'
    elif has_dna:
        mod_class = 'DNA'
    else:
        mod_class = '2OMe-2F'

    return {
        "TS_f50": context,
        "Sequence": model_seq,
        "Inhibition": round(efficacy * 100, 4),
        "Target_Seq": target_seq,
        "length": len(model_seq),
        "GC": round(gc_frac(model_seq), 4),
        "MFE": round(abs(ctx_mfe), 4),
        "start.site": 0,
        "TargetRNA": transcript,
        "TM": round(wallace_tm(model_seq), 4),
        "ASOstructure": aso_struct,
        "ASOMFE": round(aso_mfe, 4),
        "dG": round(dg, 4),
        "concentration": 10.0,
        "self_bind": round(self_bind, 4),
        "TargetGene": gene,
        "modify": mod_class,
        "Classify": 1 if efficacy >= CLASSIFY_THRESHOLD else 0,
        "open_prob": round(open_prob, 6),
        "max_open_length": round(max_open_length, 6),
        "open_pc": round(open_pc, 6),
    }

def main():
    src, dst, limit = sys.argv[1], sys.argv[2], (int(sys.argv[3]) if len(sys.argv) > 3 else None)
    with open(src, newline='') as f:
        r = csv.DictReader(f)
        rows = list(r)
    if limit:
        rows = rows[:limit]
    idx = {c: c for c in rows[0].keys()}  # DictReader: key 即列名
    out = []
    for n, row in enumerate(rows, 1):
        try:
            out.append(convert_row(row, idx))
        except Exception as e:
            print(f"[warn] row {n} failed: {e}", file=sys.stderr)
        if n % 500 == 0:
            print(f"  processed {n}/{len(rows)}", file=sys.stderr)
    with open(dst, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=OUT_COLS)
        w.writeheader()
        w.writerows(out)
    print(f"wrote {len(out)} rows -> {dst}")

if __name__ == "__main__":
    main()
