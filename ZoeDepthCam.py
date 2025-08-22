import cv2
import numpy as np
import urllib.request
import os

# --- CONFIGURAÇÃO CORRETA ---
model_url = "https://github.com/isl-org/MiDaS/releases/download/v2_1/model-small.onnx"
model_name = "model-small.onnx"

if not os.path.exists(model_name):
    print("Baixando model-small.onnx... (aguarde, ~50MB)")
    urllib.request.urlretrieve(model_url, model_name)
    print("Download completo!")

# Carrega o modelo
net = cv2.dnn.readNet(model_name)
print(f"Modelo {model_name} carregado com sucesso! ✅")

# ⚠️ FORÇA USO DA CPU (remove tentativa de CUDA)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
print("🔶 Usando CPU (CUDA não disponível nesta versão do OpenCV)")

# --- OTIMIZAÇÕES PARA CPU ---
# Reduz ainda mais a resolução para melhor performance
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)   # 📉 Reduz para 320px
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)  # 📉 Reduz para 240px
cap.set(cv2.CAP_PROP_FPS, 30)            # ⏱️  Limita para 15 FPS

print("Pressione 'q' para sair...")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # --- PRÉ-PROCESSAMENTO ---
    input_img = cv2.resize(frame, (256, 256))
    input_blob = cv2.dnn.blobFromImage(
        input_img, 
        1/255.0,
        (256, 256),
        (123.675, 116.28, 103.53),
        True,
        crop=False
    )

    # --- INFERÊNCIA ---
    net.setInput(input_blob)
    depth_map = net.forward()

    # --- PÓS-PROCESSAMENTO ---
    depth_map = depth_map[0, 0]
    depth_map = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    depth_map = cv2.applyColorMap(depth_map, cv2.COLORMAP_INFERNO)
    depth_map = cv2.resize(depth_map, (frame.shape[1], frame.shape[0]))

    # --- EXIBIÇÃO ---
    combined = np.hstack((frame, depth_map))
    cv2.imshow('Webcam | Profundidade MiDaS (CPU)', combined)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()