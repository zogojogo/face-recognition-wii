from fastapi import FastAPI, File, UploadFile
from starlette.responses import Response
import uvicorn
from src.feature_extract import embed2dict
from infer_identification import get_name_id
from infer_verification import predict_verif
from models.inception_resnet import InceptionResnetV1
import time
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--path', type=str, default='./new_data/')
args = parser.parse_args()

enrollment_path = args.path
model = InceptionResnetV1(pretrained='vggface2')
embeddings_dict = embed2dict(enrollment_path, model)

# Create Fast API
app = FastAPI()

@app.get("/")
async def index():
    return {"messages": "Open the documentations /docs or /redoc"}

@app.post("/predict_identification")
async def predict(file: UploadFile = File(...)):
    try:
        image = await file.read()
        start_time = time.time()
        conf, predicted_class = get_name_id(image, model, embeddings_dict, 'api') 
        end_time = time.time()

        return {
            "filename": str(file.filename),
            "contentype": str(file.content_type),
            "predicted id": str(predicted_class if conf > 0.75 else "Unknown"),
            "confidence": str(conf),
            "inference time": str(end_time - start_time)
        }
    except:
        return Response("Internal server error", status_code=500)

@app.post("/predict_verification")
async def predict(file: UploadFile = File(...), file_2: UploadFile = File(...)):
    try:
        image = await file.read()
        image_2 = await file_2.read()
        start_time = time.time()
        conf, prediction = predict_verif(image, image_2, model, 'api') 
        end_time = time.time()

        return {
            "prediction": str("Same Person!" if prediction == 1 else "Different Person!"),
            "confidence": str(conf),
            "inference time": str(end_time - start_time)
        }
    except:
        return Response("Internal server error", status_code=500)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)
