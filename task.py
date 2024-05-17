from fastapi import FastAPI, UploadFile, File, Request
from fastapi.responses import HTMLResponse
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np
from PIL import Image
# from prometheus_fastapi_instrumentator import Instrumentator
from prometheus_client import Counter, Gauge, generate_latest, CONTENT_TYPE_LATEST, REGISTRY
from prometheus_client.core import CollectorRegistry
import PIL
import numpy as np
import io
import os
import time
from starlette.responses import Response
import time

app = FastAPI()

REQUEST_COUNT = Counter('http_requests_total', 'Total number of HTTP requests',['client_ip'])
RT_PER_LEN_GAUGE = Gauge('rt_per_length_gauge','Total running time of the request per unit length of input string',['client_ip'])

@app.middleware("http") # middleware runs on every http request
async def add_metrics(request: Request, call_next):
    print(request.client.host)
    client_ip = request.client.host

    REQUEST_COUNT.labels(client_ip=client_ip).inc()
    
    response = await call_next(request)
    
    return response

@app.get("/metrics")
async def get_metrics():
    # Expose metrics to Prometheus
    metrics_data = generate_latest(REGISTRY)
    return Response(content=metrics_data, media_type=CONTENT_TYPE_LATEST)

# Root endpoint, returns an empty response
@app.get("/")
async def root():
    return {"message": "Hello World"}

# Endpoint to predict the digit from an uploaded image file
@app.post("/predict")
async def predict(request: Request, file: UploadFile):
    client_ip = request.client.host
    start_time = time.time()
    # Read the content of the uploaded file
    request_object_content = await file.read()
    # Open the image using PIL library
    img = Image.open(io.BytesIO(request_object_content)) 
    
    # Resize and format the image for model prediction
    resized_img = await format_image(img)

    # Convert the image data to a numpy array
    arr = np.array(resized_img)
    
    print(arr,arr.shape)

    # Flatten the image array
    flattened_image=arr.reshape(-1)
    flattened_image_list = flattened_image.tolist()
    
    # Load the trained model
    model = await load_model("/Users/nikhilanand/Prometheus_FastAPI_BDL/training_1/cp.weights.h5")
    # Predict the digit using the loaded model
    digit = await predict_digit(model,flattened_image_list)

    end_time = time.time()

    RT_PER_LEN_GAUGE.labels(client_ip=client_ip).set((end_time-start_time)/len(flattened_image_list))

    # Return the predicted digit
    return {"digit":digit,"elapsed":end_time-start_time}

# Function to predict the digit using the loaded model
async def predict_digit(model:Sequential,data_point:list)->str:
    return str(np.argmax(model(np.array(data_point).reshape(1,784))))

# Function to load the trained model
async def load_model(path:str) -> Sequential:
    # input validation
    assert os.path.exists(path)

    # Define a new Sequential model
    model2 = keras.Sequential()
    # Add layers to the model
    model2.add(layers.Dense(256, activation='sigmoid', input_shape=(784,)))
    model2.add(layers.Dense(128, activation='sigmoid'))
    model2.add(layers.Dense(10, activation='softmax'))
    # Compile the model
    model2.compile(loss='categorical_crossentropy', metrics=['accuracy'])
    # Load the weights from the specified path
    model2.load_weights(path)
    return model2

# Function to format the image for model prediction
async def format_image(img:Image):
    
    # Convert the image to grayscale
    gray_img = img.convert('L')
    # Resize the image to 28x28 pixels
    resized_img = gray_img.resize((28, 28))
    # Invert the colors of the resized image
    resized_img = PIL.ImageOps.invert(resized_img)
    # Save the resized image (for debugging purposes)
    resized_img.save("resized.jpg")
    return resized_img