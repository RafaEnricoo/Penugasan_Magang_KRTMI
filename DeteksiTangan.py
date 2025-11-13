import cv2
import mediapipe as mp

# Inisialisasi MediaPipe
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)

# Inisialisasi Kamera
cap = cv2.VideoCapture(0)

# Daftar ID landmark ujung jari (tip)
tip_ids = [4, 8, 12, 16, 20]  # ibu jari - kelingking

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Balikkan gambar agar seperti cermin
    frame = cv2.flip(frame, 1)

    # Konversi ke RGB untuk MediaPipe
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb_frame)

    finger_count = 0  # jumlah jari terangkat

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            # Gambar kerangka tangan
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Ambil semua koordinat landmark
            lm_list = []
            h, w, _ = frame.shape
            for id, lm in enumerate(hand_landmarks.landmark):
                cx, cy = int(lm.x * w), int(lm.y * h)
                lm_list.append((id, cx, cy))

            if lm_list:
                fingers = []

                # ===== IBU JARI =====
                # Periksa arah sumbu X (karena ibu jari horizontal)
                if lm_list[tip_ids[0]][1] > lm_list[tip_ids[0] - 1][1]:
                    fingers.append(1)
                else:
                    fingers.append(0)

                # ===== 4 JARI LAIN =====
                for id in range(1, 5):
                    if lm_list[tip_ids[id]][2] < lm_list[tip_ids[id] - 2][2]:
                        fingers.append(1)
                    else:
                        fingers.append(0)

                finger_count = fingers.count(1)

                # Tampilkan hasil di layar
                cv2.putText(frame, f"Jari: {finger_count}", (30, 100),
                            cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 4)

    cv2.imshow("Deteksi Jari Tangan - Polirevo", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
