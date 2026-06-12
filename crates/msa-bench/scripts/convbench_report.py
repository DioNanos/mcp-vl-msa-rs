import sys
import json, random
from collections import defaultdict
random.seed(7)

rows=[json.loads(l) for l in open(sys.argv[1] if len(sys.argv) > 1 else 'crates/msa-bench/results/2026-06-05_convbench_C1-C3b.jsonl')]
# per (qid) -> {cond: row}; abstention = id che finisce in _abs
byq=defaultdict(dict)
for r in rows: byq[r['question_id']][r['condition']]=r

def cat_of(r):
    qt=r['question_type']
    if qt.startswith('single-session'): return qt  # tenute separate
    return qt

def hit5(r): return r['first_gold_rank'] is not None and r['first_gold_rank']<5
def cov(r): return r['evidence_covered']/max(r['evidence_total'],1)

conds=['bm25-short','bm25-rich','recency','oracle-band']
cats=defaultdict(list)
absn=0
for qid,d in byq.items():
    if qid.endswith('_abs'): absn+=1; continue
    if len(d)==4: cats[cat_of(d['bm25-rich'])].append(d)

def boot_delta(pairs, f, a, b, n=1000):
    """paired bootstrap del delta mean(f(b))-mean(f(a))"""
    base=[(f(d[a]),f(d[b])) for d in pairs]
    obs=sum(y-x for x,y in base)/len(base)
    deltas=[]
    for _ in range(n):
        s=[random.choice(base) for _ in base]
        deltas.append(sum(y-x for x,y in s)/len(s))
    deltas.sort()
    return obs*100, deltas[int(n*0.025)]*100, deltas[int(n*0.975)]*100

print(f"abstention escluse: {absn}")
print(f"{'categoria':28s} {'n':>4s} | {'C1':>5s} {'C2':>5s} {'C3a':>5s} {'C3b':>5s} | delta vs C2 [CI95]")
for cat in sorted(cats, key=lambda c:-len(cats[c])):
    P=cats[cat]; n=len(P)
    r5={c: 100*sum(hit5(d[c]) for d in P)/n for c in conds}
    line=f"{cat:28s} {n:>4d} | {r5['bm25-short']:5.1f} {r5['bm25-rich']:5.1f} {r5['recency']:5.1f} {r5['oracle-band']:5.1f} |"
    for name,c in [('C2-C1','bm25-short'),('C3a-C2',None),('C3b-C2',None)]:
        pass
    d21=boot_delta(P,hit5,'bm25-short','bm25-rich')
    d32=boot_delta(P,hit5,'bm25-rich','recency')
    db2=boot_delta(P,hit5,'bm25-rich','oracle-band')
    gate='GATE' if n>=30 else 'dir.'
    print(f"{line} rich{d21[0]:+5.1f}[{d21[1]:+.0f},{d21[2]:+.0f}] rec{d32[0]:+5.1f}[{d32[1]:+.0f},{d32[2]:+.0f}] orc{db2[0]:+5.1f}[{db2[1]:+.0f},{db2[2]:+.0f}] ({gate})")

print()
print("coverage@10 (evidence-set, multi-session focus):")
for cat in ['multi-session']:
    P=cats[cat]; n=len(P)
    for c in conds:
        print(f"  {c:12s}: {100*sum(cov(d[c]) for d in P)/n:.1f}%")
