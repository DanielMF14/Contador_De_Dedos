import cv2
import mediapipe as mp

# Inicializar o detector de mãos do MediaPipe
mp_maos = mp.solutions.hands
mp_desenho = mp.solutions.drawing_utils
maos = mp_maos.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)

# Função para contar os dedos levantados
def contar_dedos(resultados):
    dedos_levantados = 0
    
    # Definição dos pontos dos dedos (polegar, indicador, médio, anelar, mínimo)
    dedos = [4, 8, 12, 16, 20]
    
    if resultados.multi_hand_landmarks:
        for mao_landmarks in resultados.multi_hand_landmarks:
            # Coordenada do ponto de referência (punho)
            y_punho = mao_landmarks.landmark[0].y
            
            for i in range(1, 5):  # Comparar dedos com o nó anterior
                if mao_landmarks.landmark[dedos[i]].y < mao_landmarks.landmark[dedos[i] - 2].y:
                    dedos_levantados += 1

            # Polegar (comparação com a palma da mão)
            if mao_landmarks.landmark[4].x > mao_landmarks.landmark[3].x:
                dedos_levantados += 1
                
    return dedos_levantados

# Captura de vídeo
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Converte para RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    resultados = maos.process(frame_rgb)

    # Desenhar os landmarks da mão
    if resultados.multi_hand_landmarks:
        for landmarks in resultados.multi_hand_landmarks:
            mp_desenho.draw_landmarks(frame, landmarks, mp_maos.HAND_CONNECTIONS)

    # Contar os dedos levantados
    dedos = contar_dedos(resultados)
    cv2.putText(frame, f"Dedos: {dedos}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Exibir a imagem
    cv2.imshow("Contador de Dedos", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
