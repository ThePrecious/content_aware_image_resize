from flask import Flask, request, jsonify, render_template
import pickle
from model_files.ml_model import run_inference
import torch
from unet import UnetGenerator
import base64
from io import BytesIO
from PIL import Image 
from urllib.request import urlopen
import numpy as np

app = Flask("resize_image")
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/resize_user_image', methods=['POST'])
def resize_user_image():
    photo_url = request.values['image_url']
    #print("Request is - ", request)
    print(photo_url)

    #load model
    model = torch.load('./model_files/best_model.pth', map_location='cpu')
    model = model.to(device, dtype=torch.float)
    model.eval()

    # constants
    resize_height = 384
    resize_width = 496

    #convert url to image numpy array
    #print("photo url:", photo_url)
    response = urlopen(photo_url)
    np_image = np.asarray(bytearray(response.read()), dtype="uint8")
    resized_image = run_inference(np_image, model)

    #convert numpy array to base64 img 
    #import pdb; pdb.set_trace()

    image = Image.fromarray(resized_image.astype('uint8'))
    buffered = BytesIO()
    image.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    #img_base64 = bytes("data:image/jpeg;base64,", encoding='utf-8') + img_str

    #response = {
    #    'output_image' : str(img_base64)
    #}
    #return str(img_base64)
    return render_template('index.html', img_str=img_str)


    

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=9696)