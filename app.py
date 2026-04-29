import streamlit as st
import joblib
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.Chem import rdFingerprintGenerator

pipeline = joblib.load("tox21_pipeline.pkl")

models = pipeline["models"]
selectors = pipeline["selectors"]
vts = pipeline["vts"]
thresholds = pipeline["thresholds"]

descriptor_names = [x[0] for x in Descriptors._descList]

morgan_gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)

def smiles_to_morgan(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return np.array(morgan_gen.GetFingerprint(mol))

def get_descriptors(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return np.array([
        getattr(Descriptors, name)(mol) if hasattr(Descriptors, name) else 0
        for name in descriptor_names
    ])

def predict(smiles):
    morgan = smiles_to_morgan(smiles)
    desc = get_descriptors(smiles)

    if morgan is None or desc is None:
        return {"error": "Invalid SMILES"}

    X = np.concatenate([morgan, desc]).reshape(1, -1)

    results = {}
    for target in models:
        X_vt = vts[target].transform(X)
        X_sel = selectors[target].transform(X_vt)
        prob = models[target].predict_proba(X_sel)[0][1]
        pred = int(prob >= thresholds[target])
        results[target] = {"prob": float(prob), "pred": pred}

    return results

st.title(" Toxicity Predictor")

smiles = st.text_input("Enter SMILES")

if st.button("Predict"):
    res = predict(smiles)

    if "error" in res:
        st.error("Invalid SMILES")
    else:
        st.json(res)