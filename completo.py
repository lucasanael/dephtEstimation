import torch
from PIL import Image, ImageDraw, ImageFont
import numpy as np
from transformers import AutoModelForDepthEstimation, AutoImageProcessor
import os
import cv2
import matplotlib.pyplot as plt
from matplotlib import colormaps
from datetime import datetime

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

# Criar pasta com data e horário atual
data_hora_atual = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
pasta_resultados = f"resultados_{data_hora_atual}"
os.makedirs(pasta_resultados, exist_ok=True)
print(f"📁 Pasta criada: {pasta_resultados}")

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

    # Obter profundidade real em metros
    depth_real = predicted_depth.cpu().numpy()

    # Para visualização (mapa de cores)
    depth_visual = (depth_real - depth_real.min()) / (depth_real.max() - depth_real.min())
    depth_visual = (depth_visual * 255).astype(np.uint8)
    depth_image = Image.fromarray(depth_visual)

    # Análise da profundidade
    print("\n" + "="*50)
    print("📊 ANÁLISE DE PROFUNDIDADE")
    print("="*50)
    print(f"📐 Formato dos dados: {depth_real.shape}")
    print(f"📏 Profundidade mínima: {depth_real.min():.3f} m")
    print(f"📏 Profundidade máxima: {depth_real.max():.3f} m")
    print(f"📊 Profundidade média: {depth_real.mean():.3f} m")
    print(f"📈 Profundidade mediana: {np.median(depth_real):.3f} m")

    # Distribuição por profundidade
    print("\n" + "="*50)
    print("📈 DISTRIBUIÇÃO POR PROFUNDIDADE")
    print("="*50)
    
    distancias = [0.5, 1.0, 1.5, 2.0, 3.0, 5.0]
    total_pixels = depth_real.size
    
    for i in range(len(distancias)):
        if i == 0:
            mascara = depth_real < distancias[i]
            texto = f"🔹 Menos de {distancias[i]}m"
        else:
            mascara = (depth_real >= distancias[i-1]) & (depth_real < distancias[i])
            texto = f"🔸 Entre {distancias[i-1]}m e {distancias[i]}m"
        
        pixels = np.sum(mascara)
        porcentagem = (pixels / total_pixels) * 100
        print(f"{texto}: {pixels:>6,d} pixels ({porcentagem:>5.1f}%)")
    
    # Acima da última distância
    mascara = depth_real >= distancias[-1]
    pixels = np.sum(mascara)
    porcentagem = (pixels / total_pixels) * 100
    print(f"🔴 Acima de {distancias[-1]}m: {pixels:>6,d} pixels ({porcentagem:>5.1f}%)")

    # Encontrar objetos próximos
    print("\n" + "="*50)
    print("🔍 OBJETOS PRÓXIMOS (< 1.5m)")
    print("="*50)
    
    mascara_proximos = depth_real < 1.5
    if np.any(mascara_proximos):
        mascara_uint8 = (mascara_proximos * 255).astype(np.uint8)
        contours, _ = cv2.findContours(mascara_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        print(f"✅ Encontrados {len(contours)} objeto(s) próximo(s)")
        
        # Criar imagem com bounding boxes
        img_with_boxes = np.array(image.copy())
        
        objetos_detectados = 0
        for i, contour in enumerate(contours):
            area = cv2.contourArea(contour)
            if area < 100:  # Ignorar objetos muito pequenos
                continue
                
            x, y, w, h = cv2.boundingRect(contour)
            
            # Calcular profundidade média do objeto
            mascara_objeto = np.zeros_like(mascara_uint8)
            cv2.drawContours(mascara_objeto, [contour], -1, 255, thickness=cv2.FILLED)
            mascara_bool = mascara_objeto == 255
            profundidade_media = np.mean(depth_real[mascara_bool])
            
            objetos_detectados += 1
            print(f"\n📦 Objeto {objetos_detectados}:")
            print(f"   📏 Área: {area:.0f} pixels")
            print(f"   📍 Bounding Box: ({x}, {y}, {w}, {h})")
            print(f"   📐 Profundidade média: {profundidade_media:.3f} m")
            print(f"   🔍 Centro: ({x + w//2}, {y + h//2})")
            
            # Desenhar bounding box na imagem
            cv2.rectangle(img_with_boxes, (x, y), (x + w, y + h), (0, 255, 0), 3)
            cv2.putText(img_with_boxes, f"Obj {objetos_detectados}", (x, y-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Marcar ponto central
            centro_x, centro_y = x + w//2, y + h//2
            cv2.circle(img_with_boxes, (centro_x, centro_y), 5, (255, 0, 0), -1)
            cv2.putText(img_with_boxes, f"{profundidade_media:.2f}m", 
                       (centro_x + 10, centro_y), cv2.FONT_HERSHEY_SIMPLEX, 
                       0.6, (255, 0, 0), 2)
        
        if objetos_detectados == 0:
            print("❌ Nenhum objeto significativo encontrado (área muito pequena)")
    else:
        print("❌ Nenhum objeto próximo encontrado")
        img_with_boxes = np.array(image.copy())

    # Salvar resultados
    nome_arquivo = os.path.basename(caminho_da_imagem)
    nome_base, extensao = os.path.splitext(nome_arquivo)
    
    # Salvar mapa de profundidade visual
    caminho_profundidade = os.path.join(pasta_resultados, f"{nome_base}_profundidade.jpg")
    depth_image.save(caminho_profundidade)
    
    # Salvar dados de profundidade
    caminho_numpy = os.path.join(pasta_resultados, f"{nome_base}_profundidade.npy")
    np.save(caminho_numpy, depth_real)
    
    # Salvar imagem com bounding boxes
    img_boxes_pil = Image.fromarray(img_with_boxes)
    caminho_boxes = os.path.join(pasta_resultados, f"{nome_base}_objetos.jpg")
    img_boxes_pil.save(caminho_boxes)
    
    # Criar mapa de calor colorido em alta qualidade
    plt.figure(figsize=(12, 10))
    plt.imshow(depth_real, cmap='jet', aspect='auto')
    plt.colorbar(label='Profundidade (metros)', shrink=0.8)
    plt.title('🎨 Mapa de Profundidade - Escala em Metros', fontsize=16, fontweight='bold')
    plt.axis('off')
    caminho_mapa_calor = os.path.join(pasta_resultados, f"{nome_base}_mapa_calor.png")
    plt.savefig(caminho_mapa_calor, dpi=300, bbox_inches='tight')
    plt.close()
    
    # Criar visualização comparativa
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
    
    # Imagem original
    ax1.imshow(np.array(image))
    ax1.set_title('🖼️ Imagem Original', fontsize=14, fontweight='bold')
    ax1.axis('off')
    
    # Mapa de profundidade
    im2 = ax2.imshow(depth_real, cmap='jet')
    ax2.set_title('🎨 Mapa de Profundidade (metros)', fontsize=14, fontweight='bold')
    ax2.axis('off')
    plt.colorbar(im2, ax=ax2, shrink=0.8)
    
    # Imagem com objetos detectados
    ax3.imshow(img_with_boxes)
    ax3.set_title('📦 Objetos Detectados', fontsize=14, fontweight='bold')
    ax3.axis('off')
    
    plt.tight_layout()
    caminho_comparativo = os.path.join(pasta_resultados, f"{nome_base}_comparativo.png")
    plt.savefig(caminho_comparativo, dpi=300, bbox_inches='tight')
    plt.close()
    
    # Criar histograma detalhado
    plt.figure(figsize=(10, 6))
    plt.hist(depth_real.flatten(), bins=50, alpha=0.7, color='skyblue', 
             edgecolor='black', linewidth=0.5)
    plt.title('📊 Distribuição de Profundidade', fontsize=16, fontweight='bold')
    plt.xlabel('Profundidade (metros)', fontsize=12)
    plt.ylabel('Frequência', fontsize=12)
    plt.grid(True, alpha=0.3)
    
    # Adicionar linhas verticais para as distâncias de referência
    colors = ['red', 'orange', 'green', 'blue', 'purple']
    for i, dist in enumerate(distancias):
        plt.axvline(x=dist, color=colors[i % len(colors)], linestyle='--', 
                   alpha=0.7, label=f'{dist}m')
    
    plt.legend()
    caminho_histograma = os.path.join(pasta_resultados, f"{nome_base}_histograma.png")
    plt.savefig(caminho_histograma, dpi=300, bbox_inches='tight')
    plt.close()
    
    # Criar relatório detalhado
    caminho_relatorio = os.path.join(pasta_resultados, f"{nome_base}_relatorio_detalhado.txt")
    with open(caminho_relatorio, 'w', encoding='utf-8') as f:
        f.write("="*60 + "\n")
        f.write("           RELATÓRIO DE ANÁLISE DE PROFUNDIDADE\n")
        f.write("="*60 + "\n\n")
        
        f.write(f"Data e hora da análise: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Arquivo analisado: {caminho_da_imagem}\n")
        f.write(f"Dimensões: {image.size}\n")
        f.write(f"Total de pixels: {total_pixels:,}\n\n")
        
        f.write("ESTATÍSTICAS GERAIS:\n")
        f.write("-" * 30 + "\n")
        f.write(f"Mínima:    {depth_real.min():.3f} m\n")
        f.write(f"Máxima:    {depth_real.max():.3f} m\n")
        f.write(f"Média:     {depth_real.mean():.3f} m\n")
        f.write(f"Mediana:   {np.median(depth_real):.3f} m\n\n")
        
        f.write("DISTRIBUIÇÃO POR PROFUNDIDADE:\n")
        f.write("-" * 40 + "\n")
        for i in range(len(distancias)):
            if i == 0:
                mascara = depth_real < distancias[i]
                texto = f"Menos de {distancias[i]}m"
            else:
                mascara = (depth_real >= distancias[i-1]) & (depth_real < distancias[i])
                texto = f"Entre {distancias[i-1]}m e {distancias[i]}m"
            
            pixels = np.sum(mascara)
            porcentagem = (pixels / total_pixels) * 100
            f.write(f"{texto:<20}: {pixels:>8,} pixels ({porcentagem:>5.1f}%)\n")
        
        mascara = depth_real >= distancias[-1]
        pixels = np.sum(mascara)
        porcentagem = (pixels / total_pixels) * 100
        f.write(f"Acima de {distancias[-1]}m{' '*(15-len(str(distancias[-1])))}: {pixels:>8,} pixels ({porcentagem:>5.1f}%)\n\n")
        
        f.write("OBJETOS PRÓXIMOS DETECTADOS:\n")
        f.write("-" * 35 + "\n")
        if objetos_detectados > 0:
            f.write(f"Total de objetos detectados: {objetos_detectados}\n")
        else:
            f.write("Nenhum objeto significativo detectado\n")
    
    # Criar arquivo de metadados com informações da execução
    caminho_metadados = os.path.join(pasta_resultados, "metadados.txt")
    with open(caminho_metadados, 'w', encoding='utf-8') as f:
        f.write(f"Data e hora: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Modelo: Intel/zoedepth-nyu-kitti\n")
        f.write(f"Dispositivo: {device}\n")
        f.write(f"Arquivo processado: {caminho_da_imagem}\n")
        f.write(f"Tempo de processamento: {datetime.now().strftime('%H:%M:%S')}\n")
    
    print(f"\n" + "="*50)
    print("💾 ARQUIVOS SALVOS")
    print("="*50)
    print(f"📊 Mapa de profundidade: {caminho_profundidade}")
    print(f"🎨 Mapa de calor: {caminho_mapa_calor}")
    print(f"📦 Imagem com objetos: {caminho_boxes}")
    print(f"🔄 Visualização comparativa: {caminho_comparativo}")
    print(f"📈 Histograma: {caminho_histograma}")
    print(f"💾 Dados numéricos: {caminho_numpy}")
    print(f"📝 Relatório detalhado: {caminho_relatorio}")
    print(f"📋 Metadados: {caminho_metadados}")
    
    print(f"\n✅ Processamento finalizado com sucesso!")
    print(f"📁 Todos os arquivos salvos em: {pasta_resultados}/")
    
    # Mostrar resumo final
    print(f"\n" + "="*50)
    print("📋 RESUMO DA ANÁLISE")
    print("="*50)
    print(f"📍 Faixa de profundidade: {depth_real.min():.2f}m - {depth_real.max():.2f}m")
    print(f"📊 Profundidade predominante: {depth_real.mean():.2f}m ± {(depth_real.std()):.2f}m")
    print(f"🔍 Objetos próximos detectados: {objetos_detectados}")
    
    # Encontrar a faixa com maior porcentagem
    porcentagens = []
    for i in range(len(distancias)):
        if i == 0:
            mascara = depth_real < distancias[i]
        else:
            mascara = (depth_real >= distancias[i-1]) & (depth_real < distancias[i])
        porcentagem = (np.sum(mascara) / total_pixels) * 100
        porcentagens.append(porcentagem)
    
    # Adicionar a faixa acima da última distância
    mascara = depth_real >= distancias[-1]
    porcentagem = (np.sum(mascara) / total_pixels) * 100
    porcentagens.append(porcentagem)
    
    max_index = np.argmax(porcentagens)
    if max_index == 0:
        faixa_maior = f"menos de {distancias[0]}m"
    elif max_index < len(distancias):
        faixa_maior = f"entre {distancias[max_index-1]}m e {distancias[max_index]}m"
    else:
        faixa_maior = f"acima de {distancias[-1]}m"
    
    print(f"📏 Maioria dos pixels ({porcentagens[max_index]:.1f}%) {faixa_maior}")
    
    # Mostrar caminho absoluto da pasta
    caminho_absoluto = os.path.abspath(pasta_resultados)
    print(f"📂 Localização: {caminho_absoluto}")