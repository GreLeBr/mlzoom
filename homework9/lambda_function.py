# import argparse
import tflite_runtime.interpreter as tflite
import numpy as np
from io import BytesIO
from urllib import request
from PIL import Image


def download_image(url):
    """ download image from url """
    with request.urlopen(url) as resp:
        buffer = resp.read()
    stream = BytesIO(buffer)
    img = Image.open(stream)
    return img

def prepare_image(img, target_size):
    """ resize img to target size """
    if img.mode != 'RGB':
        img = img.convert('RGB')
    img = img.resize(target_size, Image.Resampling.NEAREST)
    return img

def prep_input(img_url, target_size):
    """ prep and shape array """
    img = download_image (img_url)
    img_prep = prepare_image(img, target_size)
    x = np.array(img_prep)
    X = np.array([x], dtype="float32")
    X = np.multiply(X, 1./255)
    return X

def load_model(path):
    """ load tflite model """
    interpreter = tflite.Interpreter(model_path=path)
    interpreter.allocate_tensors()
    input_index = interpreter.get_input_details()[0]["index"]  
    target_size = interpreter.get_input_details()[0]["shape"][1:3]
    output_index = interpreter.get_output_details()[0]["index"]  
    return interpreter, input_index , target_size , output_index

def pred(interpreter , input, input_index, output_index ):
    """ calculate predictions """
    interpreter.set_tensor(input_index, input)
    interpreter.invoke()    
    preds = interpreter.get_tensor(output_index)
    return preds

# def main(model_path, img_url):

def main( img_url):
    
    # interpreter, input_index , target_size , output_index  = load_model("./model.tflite")
    interpreter, input_index , target_size , output_index  = load_model("dino-vs-dragon-v2.tflite") 
    input  = prep_input (img_url, target_size)
    preds = pred(interpreter, input, input_index, output_index )
    float_predictions = preds[0].tolist()
    return float_predictions
    
def lambda_handler(event, context):    
    url = event["url"]
    result = main(url)
    response = {'result': result}
    return response
    

# if __name__ == '__main__':
    
#     parser = argparse.ArgumentParser()
#     parser.add_argument(
#         "--model_path",
#         help="location of tflite model"
#     )
#     parser.add_argument(
#         "--img_path",
#         help="location of img url"
#     )
#     args = parser.parse_args()

#     main(args.model_path, args.img_path)