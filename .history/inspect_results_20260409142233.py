"""Print a clean summary of everything in semantic_mae/results/.

Run:  python inspect_results.py
"""
import json, os

HERE = os.path.dirname(os.path.abspath(__file__))
RES  = os.path.join(HERE, "results")

def load(name):
    p = os.path.join(RES, name)
    if not os.path.exists(p):
        return None
    with open(p) as f:
        return json.load(f)

def hr(ch="-", n=78): print(ch * n)

# -------------------- Teacher --------------------
teacher = load("teacher_history.json")
hr("=")
print("TEACHER (supervised ViT used for attention maps)")
hr("=")
if teacher is None:
    print("  [missing teacher_history.json]")
else:
    for e in teacher:
        print(f"  ep {e['epoch']:>2d}: train_loss={e['train_loss']:.4f} "
              f"train_acc={e['train_acc']:.4f} test_acc={e['test_acc']:.4f} "
              f"({e['time']:.1f}s)")
    print(f"  -> final teacher test acc: {teacher[-1]['test_acc']:.4f}")

# -------------------- MAE pretraining --------------------
hr("=")
print("MAE PRETRAINING (reconstruction loss on masked patches, lower = easier)")
hr("=")
print(f"  {'strategy':<16s} {'first':>10s} {'mid':>10s} {'final':>10s}  n_epochs")
for tag in ["random", "semantic_high", "semantic_low"]:
    h = load(f"mae_{tag}_history.json")
    if h is None:
        print(f"  {tag:<16s} [missing]")
        continue
    first = h[0]["loss"]
    mid   = h[len(h)//2]["loss"]
    final = h[-1]["loss"]
    print(f"  {tag:<16s} {first:>10.4f} {mid:>10.4f} {final:>10.4f}  {len(h)}")

# -------------------- Linear probe --------------------
hr("=")
print("LINEAR PROBE (frozen encoder + trained linear head on CIFAR-10)")
hr("=")
print(f"  {'strategy':<16s} {'best_acc':>10s} {'final_acc':>10s}  delta_vs_random_pp")
probes = {}
for tag in ["random", "semantic_high", "semantic_low"]:
    p = load(f"probe_{tag}.json")
    probes[tag] = p
base = probes["random"]
for tag in ["random", "semantic_high", "semantic_low"]:
    p = probes[tag]
    if p is None:
        print(f"  {tag:<16s} [missing]")
        continue
    best = 100 * p["best_test_acc"]
    final = 100 * p["final_test_acc"]
    if tag == "random" or base is None:
        delta = "     -"
    else:
        delta = f"{100*(p['best_test_acc']-base['best_test_acc']):+.2f}"
    print(f"  {tag:<16s} {best:>9.2f}% {final:>9.2f}%  {delta}")

# -------------------- Quick interpretation --------------------
hr("=")
print("QUICK INTERPRETATION")
hr("=")
losses = {}
for tag in ["random", "semantic_high", "semantic_low"]:
    h = load(f"mae_{tag}_history.json")
    if h is not None:
        losses[tag] = h[-1]["loss"]

accs = {tag: (probes[tag]["best_test_acc"] if probes[tag] else None)
        for tag in ["random", "semantic_high", "semantic_low"]}

def tick(cond): return "YES" if cond else "NO "

if all(v is not None for v in losses.values()):
    print(f"  H1: semantic_high loss > random loss?       {tick(losses['semantic_high'] > losses['random'])}  "
          f"(gap = {losses['semantic_high']-losses['random']:+.4f})")
    print(f"  H2: semantic_high loss > semantic_low loss? {tick(losses['semantic_high'] > losses['semantic_low'])}  "
          f"(gap = {losses['semantic_high']-losses['semantic_low']:+.4f})")
if all(v is not None for v in accs.values()):
    print(f"  H3: semantic_high probe >= random probe?    {tick(accs['semantic_high'] >= accs['random'])}  "
          f"(gap = {100*(accs['semantic_high']-accs['random']):+.2f} pp)")
    print(f"  H4: semantic_low probe < random probe?      {tick(accs['semantic_low'] < accs['random'])}  "
          f"(gap = {100*(accs['semantic_low']-accs['random']):+.2f} pp)")

# -------------------- Summary JSON --------------------
s = load("summary.json")
if s is not None:
    hr("=")
    print("summary.json runs:")
    for k, v in s.get("runs", {}).items():
        print(f"  {k:<16s} best={v['best_test_acc']:.4f}  final={v['final_test_acc']:.4f}")
    if "total_time_sec" in s:
        print(f"  total time: {s['total_time_sec']:.1f}s")
hr("=")
print("Done. Paste this output back so we can write the final conclusions.")