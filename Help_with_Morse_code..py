import time
import cv2
import os
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

morse_alfabesi = {
    'A': '.-', 'B': '-...', 'C': '-.-.', 'D': '-..', 'E': '.', 'F': '..-.', 'G': '--.', 'H': '....',
    'I': '..', 'J': '.---', 'K': '-.-', 'L': '.-..', 'M': '--', 'N': '-.', 'O': '---', 'P': '.--.',
    'Q': '--.-', 'R': '.-.', 'S': '...', 'T': '-', 'U': '..-', 'V': '...-', 'W': '.--', 'X': '-..-',
    'Y': '-.--', 'Z': '--..'
}

def veri_yukle(dizin):
    X, y = [], []
    for etiket, klasor in enumerate(['Close-Eyes', 'Open-Eyes']):
        yol = os.path.join(dizin, klasor)
        if os.path.exists(yol):
            for resim in os.listdir(yol):
                img = cv2.imread(os.path.join(yol, resim), cv2.IMREAD_GRAYSCALE)
                if img is not None:
                    X.append(cv2.resize(img, (64, 64)))
                    y.append(etiket)
    return np.array(X).reshape(-1, 64, 64, 1) / 255.0, np.array(y)

X, y = veri_yukle(r"C:\Users\muham\Downloads\archive (7)\mrleyedataset")

if os.path.exists("goz_modeli.keras"):
    model = load_model("goz_modeli.keras")
else:
    model = Sequential([
        Conv2D(32, (3,3), activation='relu', input_shape=(64, 64, 1)),
        MaxPooling2D(2,2),
        Flatten(),
        Dense(64, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    if len(X) > 0:
        model.fit(X, y, epochs=10, validation_split=0.2)
        model.save("goz_modeli.keras")

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True)

cap = cv2.VideoCapture(0)
durum, mors_dizisi, metin = "acik", "", ""
zaman = time.time()

while True:
    ret, frame = cap.read()
    if not ret: break
    
    h, w, _ = frame.shape
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)

    anlik_durum = "acik"

    if results.multi_face_landmarks:
        landmarks = results.multi_face_landmarks[0].landmark
        
        x_min = int(min([landmarks[33].x, landmarks[133].x, landmarks[159].x, landmarks[145].x]) * w)
        x_max = int(max([landmarks[33].x, landmarks[133].x, landmarks[159].x, landmarks[145].x]) * w)
        y_min = int(min([landmarks[33].y, landmarks[133].y, landmarks[159].y, landmarks[145].y]) * h)
        y_max = int(max([landmarks[33].y, landmarks[133].y, landmarks[159].y, landmarks[145].y]) * h)

        margin_x = int((x_max - x_min) * 0.2)
        margin_y = int((y_max - y_min) * 0.5)
        x_min = max(0, x_min - margin_x)
        x_max = min(w, x_max + margin_x)
        y_min = max(0, y_min - margin_y)
        y_max = min(h, y_max + margin_y)

        if x_max > x_min and y_max > y_min:
            gri = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            kirpilan_goz = cv2.resize(gri[y_min:y_max, x_min:x_max], (64, 64)).reshape(-1, 64, 64, 1) / 255.0
            
            tahmin = model.predict(kirpilan_goz, verbose=0)[0][0]
            anlik_durum = "kapali" if tahmin < 0.5 else "acik"

    suan = time.time()
    if anlik_durum == "kapali" and durum == "acik":
        zaman = suan
        durum = "kapali"
    elif anlik_durum == "acik" and durum == "kapali":
        sure = suan - zaman
        mors_dizisi += "-" if sure > 0.4 else "."
        zaman = suan
        durum = "acik"
            
    if durum == "acik" and mors_dizisi and (suan - zaman > 1.5):
        harf = next((k for k, v in morse_alfabesi.items() if v == mors_dizisi), "")
        metin += harf
        mors_dizisi = ""
        zaman = suan

    cv2.putText(frame, f"Mors: {mors_dizisi}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
    cv2.putText(frame, f"Metin: {metin}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow("Kamera", frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()