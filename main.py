import os
import numpy as np
import cv2
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D
from keras.optimizers import Adam
from keras.layers import MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.utils import to_categorical


train_dir = 'data/train'
val_dir = 'data/test'
train_datagen = ImageDataGenerator(rescale=1./255)
val_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(48, 48),
        batch_size=64,
        color_mode="grayscale",   
        class_mode='categorical')

validation_generator = val_datagen.flow_from_directory(
        val_dir,
        target_size=(48, 48),
        batch_size=64,
        color_mode="grayscale",  
        class_mode='categorical')

 
emotion_model = Sequential()

emotion_model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(48, 48, 1)))
emotion_model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
emotion_model.add(MaxPooling2D(pool_size=(2, 2)))
emotion_model.add(Dropout(0.25))

emotion_model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
emotion_model.add(MaxPooling2D(pool_size=(2, 2)))
emotion_model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
emotion_model.add(MaxPooling2D(pool_size=(2, 2)))
emotion_model.add(Dropout(0.25))

emotion_model.add(Flatten())
emotion_model.add(Dense(1024, activation='relu'))
emotion_model.add(Dropout(0.5))
emotion_model.add(Dense(7, activation='softmax'))

cv2.ocl.setUseOpenCL(False)

emotion_dict = {0: "angry", 1: "disgusted", 2: "fearful", 3: "happy", 4: "neutral", 5: "sad", 6: "surprised"}

# Проверка наличия файла с весами
weights_file = 'emotion_model.weights.h5'

if not os.path.exists(weights_file):
    # Если файл не существует, запускаем обучение
    emotion_model_info = emotion_model.fit(
        train_generator,
        steps_per_epoch=28709 // 64,
        epochs=50,
        validation_data=validation_generator,
        validation_steps=7178 // 64
    )
    
    # Сохраняем веса модели после обучения
    emotion_model.save_weights(weights_file)
else:
    print(f"Файл {weights_file} уже существует. Обучение пропущено.")

# Загрузка изображения
image_path = 'image.jpg'  # Укажите путь к вашему изображению
frame = cv2.imread(image_path)

if frame is None:
    raise Exception("Не удалось загрузить изображение. Проверьте путь к файлу.")

gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

# Загрузка классификатора для обнаружения лиц
bounding_box = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
if bounding_box.empty():
    raise Exception("Не удалось загрузить классификатор. Проверьте путь к файлу.")

num_faces = bounding_box.detectMultiScale(gray_frame, scaleFactor=1.3, minNeighbors=5)

for (x, y, w, h) in num_faces:
    cv2.rectangle(frame, (x, y-50), (x+w, y+h+10), (255, 0, 0), 2)
    roi_gray_frame = gray_frame[y:y + h, x:x + w]
    cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray_frame, (48, 48)), -1), 0)
    emotion_prediction = emotion_model.predict(cropped_img)
    maxindex = int(np.argmax(emotion_prediction))
    cv2.putText(frame, emotion_dict[maxindex], (x+20, y-60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
    
    

 

 # Получение текста эмоции и соответствующего стикера
emotion_text = emotion_dict[maxindex]
sticker_path = f'emojis/{emotion_text}.png'  # Путь к стикеру
sticker = cv2.imread(sticker_path, cv2.IMREAD_UNCHANGED)  # Загружаем стикер

if sticker is not None:
    # Получаем размеры стикера
    sticker_height, sticker_width = sticker.shape[:2]
    
    # Устанавливаем желаемые размеры для изменения
    w = 100  # или любое другое значение
    h = 100  # или любое другое значение

    # Изменяем размер стикера под нужный размер
    sticker = cv2.resize(sticker, (w, h))

# Определение новой позиции для стикера
sticker_x = x  # Позиция по оси X (по умолчанию совпадает с лицом)
sticker_y = y - h + 50  # Позиция по оси Y (выше лица)

# Проверка, чтобы стикер не выходил за пределы изображения
if sticker_y < 0:
    sticker_y = y + h + 10  # Если выходит за пределы, разместить ниже лица

# Наложение стикера на изображение
for c in range(0, 3):
    frame[sticker_y:sticker_y + h, sticker_x:sticker_x + w, c] = sticker[:, :, c] * (sticker[:, :, 3] / 255.0) + frame[sticker_y:sticker_y + h, sticker_x:sticker_x + w, c] * (1.0 - sticker[:, :, 3] / 255.0)



 
# Отображение результата
cv2.imshow('Emotion Recognition', frame)
cv2.waitKey(0)
cv2.destroyAllWindows()
