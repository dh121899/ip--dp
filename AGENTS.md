# Agent Notes

This is a clean reproducibility project for implementing R2T on TPC-H-derived scalar queries.

Current scope:

- Keep Task 1 as a scaffold only.
- Do not implement the full R2T algorithm until the next task asks for it.
- Keep ShiftedInverse optional and clearly marked as lightweight.
- Use Python 3.11+ and src-layout packaging.
- Do not add real TPC-H data to the repository.

Development commands on Windows:

```powershell
py -3 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
.\.venv\Scripts\python.exe -m pytest
.\.venv\Scripts\ruff.exe check .
```

