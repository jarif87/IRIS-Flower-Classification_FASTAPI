from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pathlib import Path
import pickle
import numpy as np

app = FastAPI()

# Define the path to the templates folder
templates_folder = Path(__file__).parent / "templates"

# Configure Jinja2 templates
templates = Jinja2Templates(directory=str(templates_folder))

# Mount the templates folder to serve images
app.mount("/templates", StaticFiles(directory=str(templates_folder)), name="templates")

# Load the trained model
with open("rf_model_4.pkl", "rb") as file:
    model = pickle.load(file)

# Define the index route
@app.get("/", response_class=HTMLResponse)
async def index(request: Request, prediction_text: str = ""):
    return templates.TemplateResponse("index.html", {"request": request, "prediction_text": prediction_text})

# Define the prediction route
@app.post("/predict")
async def predict(request: Request,
                  Sepal_Length: float = Form(...),
                  Sepal_Width: float = Form(...),
                  Petal_Length: float = Form(...),
                  Petal_Width: float = Form(...)):
    # Convert input values to numpy array
    input_data = np.array([[Sepal_Length, Sepal_Width, Petal_Length, Petal_Width]])
    
    # Make prediction
    prediction = model.predict(input_data)
    
    # Map numerical label to species name
    species_map = {0: "Iris Setosa", 1: "Iris Versicolor", 2: "Iris Virginica"}
    predicted_species = species_map.get(prediction[0])
    
    # Return predicted species
    return templates.TemplateResponse("index.html", {"request": request, "prediction_text": predicted_species})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
