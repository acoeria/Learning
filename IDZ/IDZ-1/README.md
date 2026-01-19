# IDZ-1 (Variant 6)

Processing a dataset using Python:
- generate CSV with 10^6 rows and 5 columns (time, func1..func4)
- compute average of column 2 (func1)
- plot func2 and func3 vs time
- plot medians of func4 in 10 blocks of 10^5 rows

## Structure
- `src/` — scripts
- `data/` — datasets (full dataset generated locally)
- `figures/` — plots
- `report/` — redacted report (no title page / personal data)

## How to run
1. Generate dataset: run `src/Генерация CSV.py`
2. Average: run `src/Среднее по колонке 2.py`
3. Plot func2/func3: run `src/График значений для колонок 3 и 4.py`
4. Plot medians: run `src/Строим график медиан.py`
