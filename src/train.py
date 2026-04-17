
import os, joblib, pandas as pd, matplotlib.pyplot as plt, seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

def train_pipeline(csv_path, model_path):
    df=pd.read_csv(csv_path)
    X=df.drop("performance", axis=1)
    y=df["performance"]
    cat=["department"]
    num=[c for c in X.columns if c not in cat]

    pre=ColumnTransformer([
        ("num", Pipeline([("imp",SimpleImputer(strategy="median")),("sc",StandardScaler())]), num),
        ("cat", Pipeline([("imp",SimpleImputer(strategy="most_frequent")),("ohe",OneHotEncoder(handle_unknown="ignore"))]), cat)
    ])

    model=Pipeline([
        ("pre", pre),
        ("clf", RandomForestClassifier(n_estimators=300, random_state=42, class_weight="balanced"))
    ])

    Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=0.2,random_state=42,stratify=y)
    model.fit(Xtr,ytr)
    pred=model.predict(Xte)
    acc=accuracy_score(yte,pred)
    report=classification_report(yte,pred)

    os.makedirs("outputs", exist_ok=True)
    with open("outputs/accuracy.txt","w") as f: f.write(str(acc))
    with open("outputs/report.txt","w") as f: f.write(report)

    cm=confusion_matrix(yte,pred)
    plt.figure(figsize=(6,4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig("outputs/confusion_matrix.png")
    plt.close()

    # feature importance
    clf=model.named_steps["clf"]
    importances=clf.feature_importances_
    names=model.named_steps["pre"].get_feature_names_out()
    fi=pd.Series(importances,index=names).sort_values(ascending=False).head(10)
    plt.figure(figsize=(8,5))
    fi.sort_values().plot(kind="barh")
    plt.title("Top Feature Importance")
    plt.tight_layout()
    plt.savefig("outputs/feature_importance.png")
    plt.close()

    joblib.dump(model, model_path)
    return {"accuracy": round(acc,4), "message":"Training complete"}
