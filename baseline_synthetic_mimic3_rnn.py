"""EECS 568 course project baseline: in-hospital mortality on PyHealth's synthetic MIMIC-III.

Environment (Python 3.11):  pip install -r requirements.txt
Data: unzip the official synthetic_mimic3_eecs568.zip from Canvas into ./data/ and verify it:
    python baseline_synthetic_mimic3_rnn.py --verify
Run the baseline for each task (10 epochs) with the three seeds required for the report:
    python baseline_synthetic_mimic3_rnn.py --task mortality --model rnn --seed 42   # then 43, 44
    python baseline_synthetic_mimic3_rnn.py --task los       --model rnn --seed 42
Both tasks use the same cohort of 45,006 admissions and the same split 36,011 / 4,495 / 4,500.
"""
import hashlib
import os
import sys
import argparse
import json
import time

import torch
from pyhealth.datasets import MIMIC3Dataset, get_dataloader, split_by_patient
from pyhealth.models import RETAIN, RNN, Transformer
from pyhealth.trainer import Trainer
from pyhealth.utils import set_seed

MODELS = {"rnn": RNN, "retain": RETAIN, "transformer": Transformer}
SPLIT_SEED = 42  # the data split is the same for every run; --seed only changes model init and training

# SHA-256 of the official course files (synthetic_mimic3_eecs568.zip on Canvas)
CHECKSUMS = {
    "ADMISSIONS.csv": "f01cf56c451e1fdf7da9a950b76bf1a6553f6e49a939b7e69fba137208670302",
    "DIAGNOSES_ICD.csv": "783cd7c41c3f1d7efadbf1c3dc09b62976abefe120f5e2c7d512035c536a8d7c",
    "ICUSTAYS.csv": "131104b63964d382c699399c42cd35ddfb8e40d121565bfa50f088739b192980",
    "PATIENTS.csv": "2dd325602aeaeffe59a818900d627ba4133ee89c7593629fc2762fe181e74403",
    "PRESCRIPTIONS.csv": "0ed0c905d5b23ec7d849aa6e60a0d0a500ab6d4d0a3cb1001b0438cdfa7cc99a",
    "PROCEDURES_ICD.csv": "31d95f9320a7f5e35e3090971d0146db12ff6e45b7d6dbd434ba8b6aa7a6f89d",
}
EXPECTED = {"samples": 45006, "train": 36011, "val": 4495, "test": 4500}
EXPECTED_POSITIVES = {"mortality": 1858, "los": 9423}


def verify(root):
    ok = True
    for name, want in CHECKSUMS.items():
        path = os.path.join(root, name)
        if not os.path.exists(path):
            print(f"MISSING  {name}"); ok = False; continue
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        got = h.hexdigest()
        print(f"{'OK      ' if got == want else 'MISMATCH'} {name}")
        ok &= got == want
    print("All files match the official course dataset." if ok else "Your copy differs from the official dataset: re-download it from Canvas.")
    return ok


LOS_DAYS = 7          # prolonged stay: strictly more than 7 days


def _cohort_samples(patient, label_fn):
    """Shared cohort rule for both tasks: one sample per admission that has
    diagnosis, procedure, and drug codes. Inputs are the admission's codes only."""
    samples = []
    for visit in patient:
        conditions = visit.get_code_list(table="DIAGNOSES_ICD")
        procedures = visit.get_code_list(table="PROCEDURES_ICD")
        drugs = visit.get_code_list(table="PRESCRIPTIONS")
        if len(conditions) * len(procedures) * len(drugs) == 0:
            continue
        samples.append({
            "visit_id": visit.visit_id,
            "patient_id": patient.patient_id,
            "conditions": [conditions],
            "procedures": [procedures],
            "drugs": [drugs],
            "label": label_fn(visit),
        })
    return samples


def in_hospital_mortality_fn(patient):
    """Task 1: does the patient die during the current admission? (HOSPITAL_EXPIRE_FLAG)"""
    return _cohort_samples(patient, lambda v: int(v.discharge_status) if v.discharge_status in (0, 1) else 0)


def prolonged_los_fn(patient):
    """Task 2: does the admission last more than LOS_DAYS days? (DISCHTIME - ADMITTIME)"""
    return _cohort_samples(patient, lambda v: int((v.discharge_time - v.encounter_time).days > LOS_DAYS))


TASKS = {"mortality": in_hospital_mortality_fn, "los": prolonged_los_fn}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="./data/synthetic_mimic3", help="folder with the six official CSV files")
    ap.add_argument("--verify", action="store_true", help="check SHA-256 of the data files and exit")
    ap.add_argument("--task", choices=TASKS, default="mortality")
    ap.add_argument("--model", choices=MODELS, default="rnn")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--dev", action="store_true", help="load only a small subset of patients for debugging")
    ap.add_argument("--output", default="./output")
    args = ap.parse_args()

    if args.verify:
        sys.exit(0 if verify(args.root) else 1)
    set_seed(args.seed)

    # STEP 1: load data
    t0 = time.time()
    base = MIMIC3Dataset(
        root=args.root,
        tables=["DIAGNOSES_ICD", "PROCEDURES_ICD", "PRESCRIPTIONS"],
        dev=args.dev,
        refresh_cache=False,
    )
    base.stat()
    print(f"[load] {time.time() - t0:.0f}s")

    # STEP 2: set task and split by patient (same split for every team: seed fixed above)
    sample_dataset = base.set_task(TASKS[args.task])
    sample_dataset.stat()
    n_pos = sum(s["label"] for s in sample_dataset.samples)
    print(f"[task] samples={len(sample_dataset)} positives={n_pos} ({n_pos / len(sample_dataset):.2%})")

    train_ds, val_ds, test_ds = split_by_patient(sample_dataset, [0.8, 0.1, 0.1], seed=SPLIT_SEED)
    train_dl = get_dataloader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_dl = get_dataloader(val_ds, batch_size=args.batch_size, shuffle=False)
    test_dl = get_dataloader(test_ds, batch_size=args.batch_size, shuffle=False)
    print(f"[split] train={len(train_ds)} val={len(val_ds)} test={len(test_ds)}")
    got = {"samples": len(sample_dataset), "train": len(train_ds), "val": len(val_ds), "test": len(test_ds)}
    exp_pos = EXPECTED_POSITIVES.get(args.task)
    if got != EXPECTED or (exp_pos is not None and n_pos != exp_pos):
        print(f"WARNING: cohort differs from the course specification. expected={EXPECTED} positives={exp_pos} got={got} positives={n_pos}")

    # STEP 3: define model
    set_seed(args.seed)
    model = MODELS[args.model](
        dataset=sample_dataset,
        feature_keys=["conditions", "procedures", "drugs"],
        label_key="label",
        mode="binary",
    )
    print(f"[model] {args.model} params={sum(p.numel() for p in model.parameters()):,}")

    # STEP 4: train (checkpoints and logs go to --output)
    trainer = Trainer(model=model, metrics=["roc_auc", "pr_auc"], output_path=args.output, exp_name=f"{args.task}_{args.model}")
    t0 = time.time()
    trainer.train(train_dataloader=train_dl, val_dataloader=val_dl, epochs=args.epochs, monitor="roc_auc")
    train_s = time.time() - t0

    # STEP 5: evaluate on the held-out test split
    metrics = trainer.evaluate(test_dl)
    result = {"task": args.task, "model": args.model, "seed": args.seed, "epochs": args.epochs, "train_seconds": round(train_s),
              "test_auroc": round(float(metrics["roc_auc"]), 4), "test_auprc": round(float(metrics["pr_auc"]), 4)}
    print(json.dumps(result, indent=1))
    with open(f"{args.output}/result_{args.task}_{args.model}_seed{args.seed}.json", "w") as f:
        json.dump(result, f, indent=1)


if __name__ == "__main__":
    main()
