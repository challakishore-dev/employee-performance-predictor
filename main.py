
import os
from src.data_generator import generate_dataset
from src.train import train_pipeline

os.makedirs("data", exist_ok=True)
os.makedirs("models", exist_ok=True)
os.makedirs("outputs", exist_ok=True)
os.makedirs("images", exist_ok=True)

df=generate_dataset(1500, seed=42)
df.to_csv("data/employee_data.csv", index=False)
print("Dataset created:", df.shape)

result=train_pipeline("data/employee_data.csv","models/model.pkl")
print(result)
print("Run dashboard using: streamlit run app.py")
