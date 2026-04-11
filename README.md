# DSCI_575_project_claudia-liauw_cynthiaagata

## Instructions

### Setup

**Prerequisites:** Python 3.9 or higher

1. Clone the repo
```bash
git clone https://github.com/UBC-MDS/DSCI_575_project_cliauwyt_cea
cd DSCI_575_project_cliauwyt_cea
```

2. Create and activate virtual environment
```bash
python -m venv env
# On Windows:
env\Scripts\activate
# On macOS/Linux:
source env/bin/activate
```

3. Install dependencies
```bash
pip install -r requirements.txt
```

4. Download the data: run the notebook `notebooks/download_data.ipynb`.
5. Build and save the indices: run the notebook `notebooks/milestone1_results.ipynb`.

### Run the app
```bash
shiny run app/app.py
```