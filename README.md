# COGS188 Project template
Fork this repo.

Once you do it is is your repo. You are free to manage the repo as you see fit, edit this README, add data files, add scripts, etc. 

Also, you are free and encouraged to share this project after the course and to add it to your portfolio. Just be sure to fork it to your GitHub at the end of the quarter!

## Setup

Everything runs in the `cogs188` conda environment (Python 3.11). Install [Miniforge](https://github.com/conda-forge/miniforge) first if you do not have conda.

```bash
# 1. Create the environment (installs requirements.txt via pip)
conda env create -f environment.yml
conda activate cogs188

# 2. Install this repo as an editable package so `import warehouse_sabotage` works everywhere
pip install -e .

# 3. Register the Jupyter kernel
python -m ipykernel install --user --name cogs188

# 4. Run the tests (no display or GPU needed)
pytest
```

If `requirements.txt` changes later, update with `conda env update -f environment.yml --prune`.

### Branch workflow

- Never commit to `main`. Branch from an up-to-date `main`: `git checkout main && git pull && git checkout -b feat/<topic>` (or `fix/`, `chore/`, `exp/`).
- Make small commits with imperative messages and run `pytest` before each one.
- Before opening a pull request: `git fetch origin && git rebase origin/main`, then push and open the PR. One teammate reviews before merging.
- Never force-push a shared branch. See `project.md` for the full project guide.
