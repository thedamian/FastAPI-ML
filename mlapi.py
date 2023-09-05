#Bring in lightweight dependencies
from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import pickle

app = FastAPI()

class ScoringItem(BaseModel): 
    YearsAtCompany: float #1, // Float Value
    EmployeeSatisfaction: float #, // Float value
    Position: str #, // Manager or Non-Manager
    Salary: int #4.0 // Ordinal 1,2,3,4,5

with open("rfmodel.pkl", "rb") as f:
    model = pickle.load(f)


@app.post("/")
async def scoring_endpoint(item: ScoringItem): 
    df = pd.DataFrame([item.dict().values()], columns=item.dict().keys())
    yhat = model.predict(df)
    return {"prediction": int(yhat)}

