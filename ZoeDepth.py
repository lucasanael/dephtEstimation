import torch
from PIL import Image
import numpy as np
from transformers import AutoModelForDepthEstimation, AutoImageProcessor
import os

# Carregar modelo e processador
print("Carregando modelo e processador...")
image_processor = AutoImageProcessor.from_pretrained("Intel/zoedepth-nyu-kitti")
model = AutoModelForDepthEstimation.from_pretrained("Intel/zoedepth-nyu-kitti")

# Mover modelo para GPU se disponível
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
print(f"Dispositivo: {device}")

# Carregar imagem LOCAL
caminho_da_imagem = "paletes/palete6.jpg"

# Criar pasta para salvar resultados
pasta_resultados = "profundidade"
os.makedirs(pasta_resultados, exist_ok=True)  # Cria a pasta se não existir

# Verificar se o arquivo existe
if not os.path.exists(caminho_da_imagem):
    print(f"Erro: Arquivo '{caminho_da_imagem}' não encontrado!")
    print("Certifique-se de que:")
    print("1. O arquivo existe no diretório")
    print("2. O caminho está correto")
    print("3. A extensão do arquivo está correta (.jpg, .png, etc.)")
else:
    print(f"Carregando imagem: {caminho_da_imagem}")
    image = Image.open(caminho_da_imagem).convert('RGB')
    print(f"Tamanho da imagem: {image.size}")

    # Processar imagem
    print("Processando imagem...")
    inputs = image_processor(image, return_tensors="pt").to(device)

    # Fazer previsão
    print("Estimando profundidade...")
    with torch.no_grad():
        outputs = model(**inputs)

    # Processar a saída
    predicted_depth = outputs.predicted_depth

    # Redimensionar para o tamanho original
    predicted_depth = torch.nn.functional.interpolate(
        predicted_depth.unsqueeze(1),
        size=(image.height, image.width),
        mode="bicubic",
        align_corners=False,
    ).squeeze()

    # Normalizar para visualização
    depth = predicted_depth.cpu().numpy()
    depth = (depth - depth.min()) / (depth.max() - depth.min())
    depth = (depth * 255).astype(np.uint8)

    # Criar imagem de profundidade
    depth_image = Image.fromarray(depth)
    
    # Mostrar resultados
    print("Processamento concluído!")
    
    # Salvar resultados na pasta imagensProcessadas
    nome_arquivo = os.path.basename(caminho_da_imagem)
    nome_base, extensao = os.path.splitext(nome_arquivo)
      
    # Salvar mapa de profundidade
    caminho_profundidade = os.path.join(pasta_resultados, f"{nome_base}_profundidade.jpg")
    depth_image.save(caminho_profundidade)
    print(f"Mapa de profundidade salvo como: {caminho_profundidade}")
    
    # Mostrar mensagem final
    print(f"\nTodos os arquivos foram salvos na pasta: {pasta_resultados}/")
    print("Processamento finalizado!")