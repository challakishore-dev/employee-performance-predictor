
import pandas as pd, numpy as np
def generate_dataset(n=1500, seed=42):
    np.random.seed(seed)
    df=pd.DataFrame({
        "age": np.random.randint(22,56,n),
        "experience": np.random.randint(1,21,n),
        "salary": np.random.randint(30000,150001,n),
        "training_hours": np.random.randint(5,101,n),
        "projects": np.random.randint(1,11,n),
        "attendance": np.random.randint(70,101,n),
        "department": np.random.choice(["IT","HR","Sales","Finance"],n)
    })
    score=(df.experience*2 + df.training_hours*0.45 + df.projects*3 + df.attendance*1.15 + (df.salary/10000)*0.25)
    df["performance"]=pd.cut(score,bins=[0,120,170,1000],labels=["Low","Medium","High"])
    return df
