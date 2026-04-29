import streamlit as st
import joblib
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.Chem import AllChem
from rdkit.Chem import rdFingerprintGenerator

pipeline = joblib.load("tox21_pipeline.pkl")
models = pipeline["models"]
selectors = pipeline["selectors"]
vts = pipeline["vts"]
thresholds = pipeline["thresholds"]
descriptor_names = [x[0] for x in Descriptors._descList]

def smiles_to_morgan(smiles, radius=2, n_bits=2048):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
    arr = np.zeros((n_bits,), dtype=int)
    from rdkit import DataStructs
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr

def get_descriptors(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    desc = []
    for name in descriptor_names:
        func = getattr(Descriptors, name)
        try:
            val = func(mol)
        except:
            val = 0
        desc.append(val)
    return np.array(desc)

def predict_toxicity(smiles):
    morgan = smiles_to_morgan(smiles)
    desc = get_descriptors(smiles)
    if morgan is None or desc is None:
        return {"error": "Invalid SMILES"}
    X = np.concatenate([morgan, desc]).reshape(1, -1)
    results = {}
    for target in models.keys():
        vt = vts[target]
        selector = selectors[target]
        model = models[target]
        threshold = thresholds[target]
        X_vt = vt.transform(X)
        X_sel = selector.transform(X_vt)
        prob = model.predict_proba(X_sel)[0][1]
        pred = int(prob >= threshold)
        results[target] = {"probability": float(prob), "prediction": pred}
    return results

st.title("🧪 Tox21 Toxicity Predictor")
smiles_input = st.text_input("Enter SMILES string:")

if st.button("Predict"):
    results = predict_toxicity(smiles_input)
    if "error" in results:
        st.error("Invalid SMILES string")
    else:
        toxic_targets = [k for k, v in results.items() if v["prediction"] == 1]
        if len(toxic_targets) == 0:
            st.success("Likely Non-Toxic")
        else:
            st.warning(f" Potential Toxicity in: {toxic_targets}")
        st.subheader("Detailed Results")
        st.json(results)