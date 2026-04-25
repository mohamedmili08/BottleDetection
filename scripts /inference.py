import cv2
from ultralytics import YOLO

# 1. تحميل الموديل الذي تم تدريبه (ستجده في runs/detect/train/weights/best.pt)
model = YOLO('../runs/detect/bottle_model/weights/best.pt')

cap = cv2.VideoCapture(0)

while cap.isOpened():
    success, frame = cap.read()
    if not success: break

    # 2. التنبؤ بالذكاء الاصطناعي
    results = model.predict(frame, conf=0.6)  # لن يظهر أي نتيجة إلا لو كان متأكداً بنسبة 60%

    # 3. رسم الصناديق والبيانات
    for r in results:
        annotated_frame = r.plot()

        # منطق إضافي: إذا اكتشف "Damaged" ارفع إنذاراً
        for box in r.boxes:
            class_id = int(box.cls[0])
            if class_id == 2:  # Damaged
                cv2.putText(annotated_frame, "WARNING: DEFECT DETECTED!", (50, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

    cv2.imshow("Smart Quality Control Simulator 2026", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()