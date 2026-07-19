# %%
import requests
import pandas as pd

# Overpass API endpoint
url = "https://overpass-api.de/api/interpreter"

query = """
[out:json][timeout:600];
area["name"="KwaZulu-Natal"]["boundary"="administrative"]->.searchArea;
(
  way["highway"]["name"](area.searchArea);
);
out center;
"""

print("Querying OpenStreetMap...")

headers = {
    "User-Agent": "NLPResearchProject/1.0 (sam.research)"
}

response = requests.post(
    url,
    data=query,
    headers=headers,
    timeout=600
)

if response.status_code != 200:
    print("Status:", response.status_code)
    print(response.text)
    exit()

data = response.json()

records = []

for element in data["elements"]:

    tags = element.get("tags", {})

    street_name = tags.get("name", "")

    if not street_name:
        continue

    lat = None
    lon = None

    if "center" in element:
        lat = element["center"]["lat"]
        lon = element["center"]["lon"]

    records.append({
        "OSM_ID": element["id"],
        "Street_Name": street_name,
        "Latitude": lat,
        "Longitude": lon
    })

df = pd.DataFrame(records)

# Remove duplicate street names
df = df.drop_duplicates(subset=["Street_Name"])

# Sort alphabetically
df = df.sort_values("Street_Name")

output_file = "KZN_Street_Names.csv"

df.to_csv(output_file, index=False)

print(f"Done! Saved {len(df)} unique street names to {output_file}")
# %%
