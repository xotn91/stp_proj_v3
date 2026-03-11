# -*- coding: utf-8 -*-
"""
Run 8 P3 combinations automatically:
- scripts: p3_1_1_K1_paired_trainer_fast.py, p3_1_2_K1_unpaired_trainer_fast.py
- scaffold_mask: on/off
- assay_mask: on/off

Each child run already writes its own run_config.log.
This batch runner adds p3_0-style batch logs and verifies per-run logs exist.
"""

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime
import pandas as pd


def compose_eval_mode(scaffold_mask_on, assay_mask_on):
    if scaffold_mask_on and assay_mask_on:
        return "distinct_scaffolds_assays"
    if scaffold_mask_on:
        return "distinct_scaffolds"
    if assay_mask_on:
        return "assays_only"
    return "all"


def default_policy_for_script(script_name):
    if "p3_1_1" in script_name:
        return "paired_sum"
    return "independent"


def read_text(path):
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def resolve_meta_file(meta_path):
    p = os.path.abspath(meta_path)
    if os.path.isdir(p):
        preferred = os.path.join(p, "final_training_meta_subset.parquet")
        if os.path.isfile(preferred):
            return preferred
        parquet_files = [x for x in os.listdir(p) if x.endswith(".parquet")]
        if len(parquet_files) == 1:
            return os.path.join(p, parquet_files[0])
        if len(parquet_files) == 0:
            raise FileNotFoundError(
                f"--meta_file points to directory without parquet file: {p}"
            )
        raise ValueError(
            f"--meta_file directory has multiple parquet files; specify one file explicitly: {p}"
        )
    if not os.path.isfile(p):
        raise FileNotFoundError(f"--meta_file not found: {p}")
    if not p.endswith(".parquet"):
        raise ValueError(f"--meta_file must be a parquet file: {p}")
    return p


def validate_pair_rule(meta_file, min_neg_per_pair=10, max_neg_per_pair=10):
    df = pd.read_parquet(meta_file, columns=["pair_id", "set_type", "target_chembl_id", "cv_fold"])
    pair_sizes = df.groupby("pair_id").size()
    pos_sizes = (
        df[df["set_type"] == "Positive"]
        .groupby("pair_id")
        .size()
        .reindex(pair_sizes.index, fill_value=0)
    )
    neg_sizes = (
        df[df["set_type"] == "Negative"]
        .groupby("pair_id")
        .size()
        .reindex(pair_sizes.index, fill_value=0)
    )
    target_nunique = df.groupby("pair_id")["target_chembl_id"].nunique()
    cv_nunique = df.groupby("pair_id")["cv_fold"].nunique()
    min_rows = 1 + int(min_neg_per_pair)
    max_rows = 1 + int(max_neg_per_pair)
    result = {
        "pairs_total": int(pair_sizes.shape[0]),
        "pair_rows_below_min_count": int((pair_sizes < min_rows).sum()),
        "pair_rows_above_max_count": int((pair_sizes > max_rows).sum()),
        "pair_pos_not_1_count": int((pos_sizes != 1).sum()),
        "pair_neg_below_min_count": int((neg_sizes < int(min_neg_per_pair)).sum()),
        "pair_neg_above_max_count": int((neg_sizes > int(max_neg_per_pair)).sum()),
        "pair_target_mismatch_count": int((target_nunique != 1).sum()),
        "pair_cv_leakage_count": int((cv_nunique != 1).sum()),
    }
    if (
        result["pair_rows_below_min_count"] > 0
        or result["pair_rows_above_max_count"] > 0
        or result["pair_pos_not_1_count"] > 0
        or result["pair_neg_below_min_count"] > 0
        or result["pair_neg_above_max_count"] > 0
        or result["pair_target_mismatch_count"] > 0
        or result["pair_cv_leakage_count"] > 0
    ):
        raise ValueError(
            "meta_file violates required pair rule "
            f"(1 positive + {int(min_neg_per_pair)}~{int(max_neg_per_pair)} negative per pair_id): {result}"
        )
    return result


def main():
    parser = argparse.ArgumentParser(description="Run 8 mask combinations for P3 scripts")
    parser.add_argument("--python_bin", default=sys.executable)
    parser.add_argument("--script_dir", default=os.path.dirname(os.path.abspath(__file__)))
    parser.add_argument(
        "--scripts",
        default="p3_1_1_K1_paired_trainer_fast.py,p3_1_2_K1_unpaired_trainer_fast.py",
        help="Comma-separated script file names to run.",
    )
    parser.add_argument("--out_dir", default="features_store")
    parser.add_argument("--meta_file", default="features_store/final_training_meta.parquet")
    parser.add_argument("--fp2_memmap", default="features_store/fp2_aligned.memmap")
    parser.add_argument("--es5d_memmap", default=None)
    parser.add_argument("--k_mode", type=int, default=1)
    parser.add_argument(
        "--k_modes",
        default=None,
        help="Comma-separated K values (e.g., 1,5). If set, overrides --k_mode.",
    )
    parser.add_argument("--c_reg", type=float, default=10.0)
    parser.add_argument("--min_neg_per_pair", type=int, default=7)
    parser.add_argument("--max_neg_per_pair", type=int, default=10)
    parser.add_argument("--cutoff_year", type=int, default=2023)
    parser.add_argument("--batch_size", type=int, default=4096)
    parser.add_argument("--chunk_3d_b", type=int, default=128)
    parser.add_argument("--disable_tf32", action="store_true")
    parser.add_argument("--stop_on_error", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument(
        "--max_parallel_jobs",
        type=int,
        default=1,
        help="How many jobs to run at once. Use 2 to utilize two GPUs.",
    )
    parser.add_argument(
        "--gpu_ids",
        default="0,1",
        help="Comma-separated GPU ids used for round-robin job assignment.",
    )
    parser.add_argument(
        "--cpu_threads_per_job",
        type=int,
        default=None,
        help="OMP/MKL threads per job. Default: cpu_count // max_parallel_jobs.",
    )
    parser.add_argument(
        "--max_jobs",
        type=int,
        default=None,
        help="Run only first N expanded jobs (for quick checks).",
    )
    args = parser.parse_args()
    args.meta_file = resolve_meta_file(args.meta_file)
    pair_rule_check = validate_pair_rule(
        args.meta_file,
        min_neg_per_pair=args.min_neg_per_pair,
        max_neg_per_pair=args.max_neg_per_pair,
    )
    print(f">> Pair-rule preflight passed: {pair_rule_check}")
    if not args.es5d_memmap:
        args.es5d_memmap = "features_store/es5d_db_k20.memmap"

    if args.k_modes:
        k_modes = [int(x.strip()) for x in args.k_modes.split(",") if x.strip()]
        if not k_modes:
            raise ValueError("No valid k_modes parsed from --k_modes")
    else:
        k_modes = [args.k_mode]

    started = datetime.now()
    batch_root = os.path.join(
        args.out_dir, f"p3_0_batch_mask_grid_{started.strftime('%Y%m%d_%H%M%S')}"
    )
    os.makedirs(batch_root, exist_ok=True)
    logs_dir = os.path.join(batch_root, "batch_logs")
    os.makedirs(logs_dir, exist_ok=True)

    scripts = [s.strip() for s in args.scripts.split(",") if s.strip()]
    if not scripts:
        raise ValueError("No scripts provided via --scripts")
    mask_options = [("on", "off"), ("off", "on"), ("on", "on"), ("off", "off")]

    jobs = []
    for script_name in scripts:
        for scaffold_mask, assay_mask in mask_options:
            jobs.append(
                {
                    "script_name": script_name,
                    "scaffold_mask": scaffold_mask,
                    "assay_mask": assay_mask,
                }
            )

    batch_info = {
        "script": os.path.abspath(__file__),
        "start_time": started.isoformat(),
        "batch_root": batch_root,
        "dry_run": bool(args.dry_run),
        "global_options": {
            "scripts": scripts,
            "meta_file": args.meta_file,
            "fp2_memmap": args.fp2_memmap,
            "es5d_memmap": args.es5d_memmap,
            "k_modes": k_modes,
            "c_reg": args.c_reg,
            "min_neg_per_pair": args.min_neg_per_pair,
            "max_neg_per_pair": args.max_neg_per_pair,
            "cutoff_year": args.cutoff_year,
            "batch_size": args.batch_size,
            "chunk_3d_b": args.chunk_3d_b,
            "disable_tf32": bool(args.disable_tf32),
            "max_parallel_jobs": args.max_parallel_jobs,
            "gpu_ids": args.gpu_ids,
            "cpu_threads_per_job": args.cpu_threads_per_job,
            "max_jobs": args.max_jobs,
        },
        "pair_rule_check": pair_rule_check,
        "planned_jobs": len(jobs),
        "jobs": [],
    }

    c_reg_tag = str(args.c_reg).replace(".", "p")
    expanded_jobs = []
    for k_mode in k_modes:
        for job in jobs:
            copied = dict(job)
            copied["k_mode"] = int(k_mode)
            expanded_jobs.append(copied)

    if args.max_jobs is not None:
        expanded_jobs = expanded_jobs[: args.max_jobs]

    batch_info["planned_jobs"] = len(expanded_jobs)
    gpu_ids = [g.strip() for g in str(args.gpu_ids).split(",") if g.strip()]
    if not gpu_ids:
        gpu_ids = ["0"]
    if args.cpu_threads_per_job is None:
        cpu_total = os.cpu_count() or 1
        cpu_threads_per_job = max(1, cpu_total // max(1, args.max_parallel_jobs))
    else:
        cpu_threads_per_job = max(1, int(args.cpu_threads_per_job))

    active = []
    next_idx = 0
    terminate_requested = False
    total_jobs = len(expanded_jobs)

    def _prepare_job(idx_1based, job):
        script_path = os.path.join(args.script_dir, job["script_name"])
        if not os.path.exists(script_path):
            raise FileNotFoundError(f"Script not found: {script_path}")

        scaffold_on = job["scaffold_mask"] == "on"
        assay_on = job["assay_mask"] == "on"
        eval_mode = compose_eval_mode(scaffold_on, assay_on)
        policy = default_policy_for_script(job["script_name"])
        stem = os.path.splitext(job["script_name"])[0]
        expected_run_dir_name = (
            f"{stem}__K{job['k_mode']}__{policy}__{eval_mode}__C{c_reg_tag}__Y{args.cutoff_year}"
        )
        expected_run_dir = os.path.join(batch_root, expected_run_dir_name)
        expected_run_log = os.path.join(expected_run_dir, "run_config.log")

        cmd = [
            args.python_bin,
            script_path,
            "--meta_file",
            args.meta_file,
            "--fp2_memmap",
            args.fp2_memmap,
            "--out_dir",
            batch_root,
            "--k_mode",
            str(job["k_mode"]),
                        "--c_reg",
                        str(args.c_reg),
                        "--min_neg_per_pair",
                        str(args.min_neg_per_pair),
                        "--max_neg_per_pair",
                        str(args.max_neg_per_pair),
                        "--cutoff_year",
                        str(args.cutoff_year),
            "--batch_size",
            str(args.batch_size),
            "--chunk_3d_b",
            str(args.chunk_3d_b),
            "--scaffold_mask",
            job["scaffold_mask"],
            "--assay_mask",
            job["assay_mask"],
        ]
        if args.disable_tf32:
            cmd.append("--disable_tf32")
        cmd.extend(["--es5d_memmap", args.es5d_memmap])

        return {
            "job_index": idx_1based,
            "script": script_path,
            "k_mode": int(job["k_mode"]),
            "scaffold_mask": job["scaffold_mask"],
            "assay_mask": job["assay_mask"],
            "eval_mode": eval_mode,
            "expected_run_dir": expected_run_dir,
            "expected_run_log": expected_run_log,
            "command": cmd,
        }

    if args.dry_run:
        for idx, job in enumerate(expanded_jobs, start=1):
            job_rec = _prepare_job(idx, job)
            print(
                f"[{idx}/{total_jobs}] {job['script_name']} "
                f"K={job['k_mode']} scaffold={job['scaffold_mask']} assay={job['assay_mask']}"
            )
            job_rec["start_time"] = datetime.now().isoformat()
            job_rec["status"] = "dry_run_skipped"
            job_rec["returncode"] = None
            batch_info["jobs"].append(job_rec)
    else:
        while (next_idx < total_jobs or active) and not terminate_requested:
            while next_idx < total_jobs and len(active) < max(1, args.max_parallel_jobs):
                idx_1 = next_idx + 1
                job = expanded_jobs[next_idx]
                job_rec = _prepare_job(idx_1, job)
                assigned_gpu = gpu_ids[next_idx % len(gpu_ids)]
                job_rec["assigned_gpu"] = assigned_gpu
                job_rec["cpu_threads_per_job"] = cpu_threads_per_job
                print(
                    f"[{idx_1}/{total_jobs}] launch {job['script_name']} "
                    f"K={job['k_mode']} scaffold={job['scaffold_mask']} assay={job['assay_mask']} "
                    f"gpu={assigned_gpu}"
                )

                out_log = os.path.join(logs_dir, f"job_{idx_1:02d}.stdout.log")
                err_log = os.path.join(logs_dir, f"job_{idx_1:02d}.stderr.log")
                out_f = open(out_log, "w", encoding="utf-8")
                err_f = open(err_log, "w", encoding="utf-8")
                job_rec["stdout_log"] = out_log
                job_rec["stderr_log"] = err_log
                start_one = datetime.now()
                job_rec["start_time"] = start_one.isoformat()

                env = os.environ.copy()
                env["CUDA_VISIBLE_DEVICES"] = str(assigned_gpu)
                env["OMP_NUM_THREADS"] = str(cpu_threads_per_job)
                env["MKL_NUM_THREADS"] = str(cpu_threads_per_job)
                env["NUMEXPR_NUM_THREADS"] = str(cpu_threads_per_job)

                proc = subprocess.Popen(
                    job_rec["command"],
                    stdout=out_f,
                    stderr=err_f,
                    text=True,
                    env=env,
                )
                active.append(
                    {
                        "proc": proc,
                        "job_rec": job_rec,
                        "start": start_one,
                        "out_f": out_f,
                        "err_f": err_f,
                    }
                )
                next_idx += 1

            still_active = []
            for item in active:
                ret = item["proc"].poll()
                if ret is None:
                    still_active.append(item)
                    continue

                item["out_f"].close()
                item["err_f"].close()
                end_one = datetime.now()
                job_rec = item["job_rec"]
                job_rec["end_time"] = end_one.isoformat()
                job_rec["duration_sec"] = (end_one - item["start"]).total_seconds()
                job_rec["returncode"] = int(ret)
                job_rec["status"] = "success" if ret == 0 else "failed"

                run_dir_exists = os.path.isdir(job_rec["expected_run_dir"])
                run_log_exists = os.path.isfile(job_rec["expected_run_log"])
                job_rec["run_dir_exists"] = run_dir_exists
                job_rec["run_log_exists"] = run_log_exists

                if run_log_exists:
                    rc_text = read_text(job_rec["expected_run_log"])
                    job_rec["run_log_has_scaffold_mask"] = "scaffold_mask=" in rc_text
                    job_rec["run_log_has_assay_mask"] = "assay_mask=" in rc_text
                    job_rec["run_log_has_eval_mode"] = "eval_mode=" in rc_text
                    job_rec["run_log_status_success"] = "status=success" in rc_text
                else:
                    job_rec["run_log_has_scaffold_mask"] = False
                    job_rec["run_log_has_assay_mask"] = False
                    job_rec["run_log_has_eval_mode"] = False
                    job_rec["run_log_status_success"] = False

                batch_info["jobs"].append(job_rec)
                print(
                    f"[{job_rec['job_index']}/{total_jobs}] done status={job_rec['status']} "
                    f"rc={job_rec['returncode']}"
                )

                if ret != 0 and args.stop_on_error:
                    terminate_requested = True

            active = still_active

            if terminate_requested and active:
                for item in active:
                    try:
                        item["proc"].terminate()
                    except Exception:
                        pass
                # wait briefly and mark terminated jobs
                time.sleep(0.2)
                for item in active:
                    try:
                        item["proc"].wait(timeout=1)
                    except Exception:
                        try:
                            item["proc"].kill()
                        except Exception:
                            pass
                    item["out_f"].close()
                    item["err_f"].close()
                    end_one = datetime.now()
                    job_rec = item["job_rec"]
                    job_rec["end_time"] = end_one.isoformat()
                    job_rec["duration_sec"] = (end_one - item["start"]).total_seconds()
                    job_rec["returncode"] = None
                    job_rec["status"] = "terminated_due_to_stop_on_error"
                    batch_info["jobs"].append(job_rec)
                active = []

            if active:
                time.sleep(0.5)

    ended = datetime.now()
    batch_info["end_time"] = ended.isoformat()
    batch_info["duration_sec"] = (ended - started).total_seconds()
    batch_info["success_jobs"] = sum(1 for j in batch_info["jobs"] if j.get("status") == "success")
    batch_info["failed_jobs"] = sum(1 for j in batch_info["jobs"] if j.get("status") == "failed")
    batch_info["dry_run_jobs"] = sum(1 for j in batch_info["jobs"] if j.get("status") == "dry_run_skipped")

    json_path = os.path.join(batch_root, "batch_run_log.json")
    txt_path = os.path.join(batch_root, "batch_run_log.txt")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(batch_info, f, indent=2)

    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(f"start_time={batch_info['start_time']}\n")
        f.write(f"end_time={batch_info['end_time']}\n")
        f.write(f"duration_sec={batch_info['duration_sec']:.3f}\n")
        f.write(f"planned_jobs={batch_info['planned_jobs']}\n")
        f.write(f"success_jobs={batch_info['success_jobs']}\n")
        f.write(f"failed_jobs={batch_info['failed_jobs']}\n")
        f.write(f"dry_run_jobs={batch_info['dry_run_jobs']}\n")
        f.write(f"batch_root={batch_root}\n")

    print(">> Batch completed")
    print(f"   - batch root: {batch_root}")
    print(f"   - batch log json: {json_path}")
    print(f"   - batch log txt: {txt_path}")


if __name__ == "__main__":
    main()
