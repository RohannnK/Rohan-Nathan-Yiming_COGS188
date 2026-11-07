@project.md

- Never commit to `main`; work on `feat/`, `fix/`, `chore/` or `exp/` branches and open a PR. Never force-push.
- Run every command in the `cogs188` conda env (`conda run -n cogs188 ...`).
- Run `pytest` before every commit; it must pass and stay under 30 s.
- Scripts stay thin (argparse + config); logic lives in `src/warehouse_sabotage/`.
- All randomness goes through `numpy.random.Generator` objects seeded from configs; no global `np.random`.
- Check rware behaviour against the installed source, not memory; fix project.md section 4 if it is wrong.
- British spelling and no em dashes in README and results notes.
