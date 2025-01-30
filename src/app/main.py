import time
import logging
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, field_validator
from app.models.predict import Prediction, __version__

app = FastAPI()


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataInput(BaseModel):
    weight: float
    height: float
    gender: str

    @field_validator("gender")
    def gender_must_be_male_or_female(cls, v):
        if v.lower() not in {"male", "female"}:
            raise ValueError('Gender must be either "male" or "female"')
        return v.lower()


class PredictionOut(BaseModel):
    prediction: str


@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    start_time = time.perf_counter()
    response = await call_next(request)
    process_time = time.perf_counter() - start_time
    response.headers["X-Process-Time"] = str(process_time)
    return response


@app.get("/")
def home():
    return {"Health": "OK", "Version": __version__}


@app.post("/predict", response_model=PredictionOut)
async def predict(payload: DataInput):
    try:
        model_name = f"BMIModel-Quantized-{__version__}.tflite"
        prediction_instance = Prediction(model_name)
        prediction = prediction_instance.make_prediction(
            payload.weight, payload.height, payload.gender
        )
        return {"prediction": str(prediction)}
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(
            status_code=500, detail="An error occurred while making the prediction."
        )
