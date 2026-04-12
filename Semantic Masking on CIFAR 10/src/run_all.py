"""End-to-end pipeline: teacher -> MAE(random) -> MAE(semantic_high) -> MAE(semantic_low)
-> linear probes -> visualizations -> results summary.

Use --smoke for a very short run that proves the code works (few epochs, tiny subset).
Use defaults for the real experiment.
"""
import os, json, argparse, time
from config import get_config
from train_teacher import train_teacher
from train_mae import pretrain
from linear_probe import linear_probe
import visualize


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help="Tiny-subset, few-epoch run for testing")
    ap.add_argument("--teacher_epochs", type=int, default=None)
    ap.add_argument("--pretrain_epochs", type=int, default=None)
    ap.add_argument("--probe_epochs", type=int, default=None)
    ap.add_argument("--skip_semantic_low", action="store_true")
    args = ap.parse_args()

    cfg = get_config()
    if args.smoke:
        cfg.batch_size = 64
        cfg.num_workers = 0
        cfg.teacher_epochs = 1
        cfg.pretrain_epochs = 2
        cfg.probe_epochs = 2
        cfg.warmup_epochs = 0
        subset = 512
    else:
        subset = None
    if args.teacher_epochs: cfg.teacher_epochs = args.teacher_epochs
    if args.pretrain_epochs: cfg.pretrain_epochs = args.pretrain_epochs
    if args.probe_epochs: cfg.probe_epochs = args.probe_epochs

    os.makedirs(cfg.ckpt_dir, exist_ok=True)
    os.makedirs(cfg.results_dir, exist_ok=True)

    summary = {"config": {k: v for k, v in cfg.__dict__.items()}, "runs": {}}

    t0 = time.time()
    print("=" * 60); print("STEP 1: train teacher ViT classifier"); print("=" * 60)
    train_teacher(cfg, subset=subset)

    strategies = ["random", "semantic_high"]
    if not args.skip_semantic_low:
        strategies.append("semantic_low")

    for s in strategies:
        print("=" * 60); print(f"STEP 2: pretrain MAE ({s})"); print("=" * 60)
        pretrain(cfg, s, subset=subset, tag=s)

        print("=" * 60); print(f"STEP 3: linear probe ({s})"); print("=" * 60)
        probe = linear_probe(cfg, tag=s, subset=subset)
        summary["runs"][s] = {"best_test_acc": probe["best_test_acc"],
                              "final_test_acc": probe["final_test_acc"]}

    print("=" * 60); print("STEP 4: visualizations"); print("=" * 60)
    try:
        visualize.run(cfg, n=6)
    except Exception as e:
        print(f"[visualize] skipped due to: {e}")

    summary["total_time_sec"] = time.time() - t0
    with open(os.path.join(cfg.results_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print("\nFINAL SUMMARY:")
    for s, r in summary["runs"].items():
        print(f"  {s:<15s} best={r['best_test_acc']:.4f}  final={r['final_test_acc']:.4f}")
    print(f"total time: {summary['total_time_sec']:.1f}s")

if __name__ == "__main__":
    main()
