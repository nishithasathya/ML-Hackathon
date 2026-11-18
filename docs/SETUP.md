# Environment setup

Tested with Python 3.13 on Linux. The code uses Python 3.9+ syntax only and
plain, widely available libraries (numpy, pandas, scipy, matplotlib, folium, pytest).

## macOS / Linux

```bash
cd ~/Documents/ML-Hackathon
```
```bash
python3 -m venv .venv
```
```bash
source .venv/bin/activate
```
```bash
python -m pip install --upgrade pip
```
```bash
pip install -r requirements.txt
```

## Windows (PowerShell)

```
py -3 -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Dataset location

Either copy the folders:

```bash
cp -R "$HOME/Documents/ML LAB/HACKATHON/Student folder/train" "$HOME/Documents/ML LAB/HACKATHON/Student folder/student_test" data/raw/
```

or point the code to them without copying:

```bash
export PATHSNAP_DATA_ROOT="$HOME/Documents/ML LAB/HACKATHON/Student folder"
```

Check:

```bash
python -c "from src.common.io import list_trips; print(len(list_trips('train')), len(list_trips('student_test')))"
```

Expected output: `80 10`.

## Common problems

| Symptom | Fix |
|---|---|
| `ModuleNotFoundError: No module named 'src'` | Run commands from the repository root with `python -m ...` |
| `FileNotFoundError: Partition folder not found` | Dataset not in `data/raw/`, or `PATHSNAP_DATA_ROOT` not set in this terminal |
| Folium map is blank | The browser needs internet to load Leaflet and the map tiles from CDNs |
| `ValueError: Partition '...' is not permitted` | Intended: only `train` and `student_test` are allowed |
