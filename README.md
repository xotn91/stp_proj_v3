# stp_proj_v3

SwissTargetPrediction-style modeling project built on ChEMBL 36.

## Working Paths

- Windows: `D:\stp_proj_v3`
- WSL: `/mnt/d/stp_proj_v3`

## Main Directories

- `script/`: active scripts to modify and run
- `old_scripts/`: reference implementations, do not modify for current P3 work
- `features_store/`: generated features, memmaps, logs, training outputs
- `results/`: summaries and result review files
- `blast_work/`: paralog and target homology related materials

## Current Git Policy

- Track code and markdown documentation
- Do not track large generated artifacts such as `parquet`, `memmap`, logs, and bulk result exports
- Use small focused commits

## Recommended Workflow

```bash
cd /mnt/d/stp_proj_v3
git status
git checkout -b <branch-name>
git add <files>
git commit -m "Describe the change"
git push -u origin <branch-name>
```

## P3 Notes

- Modify only files in `script/` for current P3 changes
- Keep pair-based governance intact during sampling, split, validation, and training
- ES5D conformer count is fixed at 20

## Remote

- GitHub: `git@github.com:xotn91/stp_proj_v3.git`
