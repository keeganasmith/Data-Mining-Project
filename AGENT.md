# Repository guidance for future tasks

## Dataset column assumptions (important)
- Do **not** invent or assume columns that are not present in the dataset.
- In particular, there is **no** `winner_rank` column in this project dataset.
- Ranking-related data exists in multiple specific columns (for example, `PlayerTeam2.SglRollRank`).
- All matches in this dataset are **singles** matches, so prefer singles ranking fields and ignore doubles ranking fields unless a task explicitly asks for doubles analysis.

## Verification requirement
- Before using any column in analysis or code, first verify the exact column names from the loaded dataframe/schema (e.g., `df.columns`) and use only confirmed names.
