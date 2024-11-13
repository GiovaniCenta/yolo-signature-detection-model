from flask import Flask, request, jsonify
import os
import torch
import shutil
import warnings
from contextlib import redirect_stdout
import io

### WINDOWS ONLY - YOLO COMPATIBILITY
# Adjust pathlib for compatibility
import pathlib
temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath

#from flask import Flask, request, jsonify
import os
import torch
import shutil
import warnings
from contextlib import redirect_stdout
import io

# Suprimir warnings
warnings.filterwarnings("ignore", category=FutureWarning)

# Função para carregar o modelo
def load_model(model_path):
    with io.StringIO() as buf, redirect_stdout(buf):
        model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path, force_reload=False)
    return model

# Função para fazer a predição e retornar o valor de confiança
def make_prediction(model, image_path, confidence_threshold=0.94):
    results = model(image_path)
    detected = False
    highest_conf = 0.0  # Inicializa o maior valor de confiança

    for *box, conf, cls in results.xyxy[0]:
        conf = float(conf)  # Converte para float se não for
        if conf > confidence_threshold:
            detected = True
        highest_conf = max(highest_conf, conf)  # Atualiza o maior valor de confiança encontrado

    return detected, highest_conf, results

# Inicializa o Flask
app = Flask(__name__)

# Carregar o modelo uma vez no início
model_path = 'runs/train/exp24/weights/best.pt'  # Atualize com o caminho correto do modelo
model = load_model(model_path)
save_predictions_path = './prediction_result'
if not os.path.exists(save_predictions_path):
    os.makedirs(save_predictions_path)

# Rota para fazer a predição
@app.route('/predict', methods=['POST'])
def predict():
    # Verifica se o arquivo foi enviado na requisição
    if 'image' not in request.files:
        return jsonify({"error": "No image file provided"}), 400
    
    image_file = request.files['image']
    image_path = os.path.join(save_predictions_path, image_file.filename)
    image_file.save(image_path)
    
    # Faz a predição
    confidence_threshold = 0.94  # Defina o threshold aqui, se necessário ajustar
    detected, highest_conf, results = make_prediction(model, image_path, confidence_threshold)

    # Salva a imagem com uma indicação de "detected" no nome
    new_filename = f"{os.path.basename(image_path).split('.')[0]}"
    results.save(save_dir=os.path.join(save_predictions_path, new_filename))
    print(f"Prediction saved as {new_filename}.")

    # Retorna o resultado da predição
    return jsonify({
        "filename": new_filename,
        "prediction": "signature" if detected else "no_signature",
        "prediction_confidence_threshold": confidence_threshold,
        "prediction_confidence": round(float(highest_conf), 3)  # Converte para float
    })

# Inicializa o servidor Flask
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
