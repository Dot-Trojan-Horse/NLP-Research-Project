# %%
import pandas as pd
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent

input_file = project_root / "output" / "KZN_Street_Names_CLEAN.csv"
output_file = project_root / "output" / "KZN_Street_Names_Tagged.csv"

df = pd.read_csv(input_file)

# Common isiZulu prefixes

prefixes = (

    "u",

    "e",

    "kwa",

    "ama",

    "izi",

    "in",

    "isi",

    "um",

    "imi"

)

df["Likely_IsiZulu"] = df["Street_Name"].str.lower().str.startswith(prefixes)

df.to_csv(output_file, index=False)

print(df["Likely_IsiZulu"].value_counts())

print("Finished.")