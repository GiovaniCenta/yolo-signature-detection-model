import requests

# Defina o URL do serviço (API)
url = "http://127.0.0.1:5000/predict"

# Caminho para a imagem que você deseja enviar
image_path = "12_1.jpeg"

# Abra o arquivo de imagem e faça a requisição
with open(image_path, "rb") as image_file:
    files = {"image": image_file}
    response = requests.post(url, files=files)

# Verifique se a requisição foi bem-sucedida e imprima a resposta
if response.status_code == 200:
    print("Resposta da API:", response.json())
else:
    print(f"Erro {response.status_code}: {response.text}")
