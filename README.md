# HBB Variant Pathogenicity Predictor (prototype)

## Run in VS Code
1. Install Python 3.10+ and VS Code (with the "Python" extension).
2. File > Open Folder... > select `hbb_predictor`.
3. Terminal > New Terminal, then:
   - Windows: `python -m venv venv` then `venv\Scripts\activate`
   - macOS/Linux: `python3 -m venv venv` then `source venv/bin/activate`
4. `pip install -r requirements.txt`
5. `python app.py`
6. Open http://127.0.0.1:5000
7. Optional: put your college logo at `static/logo.png`.

## Replace the synthetic data with real data
Edit `build_training_data()` in `model.py`: download HBB variants from ClinVar
(https://www.ncbi.nlm.nih.gov/clinvar/), keep Pathogenic/Likely pathogenic vs
Benign/Likely benign, and build the same 5 features per variant.
Replace `conservation()` with real scores (phyloP / ConSurf / multi-species alignment).
