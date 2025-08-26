import pandas as pd 
import json 

df = pd.read_csv("eval_dataset.csv")
records = df.to_dict(orient="records")
with open("eval_dataset.jsonl", "w") as f:
    for r in records:
        f.write(json.dumps(r) + "\n")
print("Convered CSV to JSON")