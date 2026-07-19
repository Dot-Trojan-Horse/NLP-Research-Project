import pandas as pd
from pathlib import Path

project = Path(__file__).resolve().parent.parent

input_file = project / "data" / "KZN_Street_Names_VERIFIED.csv"
output_file = project / "output" / "Navoice_Upload.xlsx"

print("Loading dataset...")

df = pd.read_csv(input_file)

# Rename columns
df = df.rename(columns={
    "Street_Name": "Street Name"
})

# Create upload format

navoice = pd.DataFrame()

navoice["ID"] = range(1, len(df) + 1)

navoice["Street Name"] = df["Street Name"]

navoice["Language"] = "isiZulu"

# Until supervisor allocates regions
navoice["Region"] = "eThekwini"

navoice["System Output"] = ""

navoice["Ground Truth"] = ""

navoice.to_excel(output_file, index=False)

print()

print("Finished!")

print(output_file)