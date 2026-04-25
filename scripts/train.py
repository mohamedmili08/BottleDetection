from ultralytics import YOLO

def train_model():
    # تحميل أحدث موديل خفيف لعام 2026 (YOLO26n هو الأسرع للكاميرات)
    model = YOLO('yolo26n.pt')

    # بدء التدريب
    model.train(
        data='dataset/data.yaml',
        epochs=100,      # عدد مرات التكرار
        imgsz=640,       # حجم الصورة
        batch=16,        # يعتمد على قوة كرت الشاشة عندك
        name='bottle_model'
    )
#cmment 
if __name__ == '__main__':
    train_model()


