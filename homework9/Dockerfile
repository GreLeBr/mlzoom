FROM svizor42/zoomcamp-dino-dragon-lambda:v2

COPY lambda_function.py .
COPY requirements.txt .

RUN pip install --upgrade pip
RUN pip install -r requirements.txt
RUN pip install https://github.com/alexeygrigorev/tflite-aws-lambda/raw/main/tflite/tflite_runtime-2.7.0-cp39-cp39-linux_x86_64.whl

CMD [ "lambda_function.lambda_handler" ]
