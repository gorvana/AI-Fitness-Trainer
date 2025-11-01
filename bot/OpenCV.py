import cv2
import os
from tqdm import tqdm
import glob
import logging
import time

logger = logging.getLogger(__name__)

def save_frames(local_file_path: str):
    time.sleep(5)
    video_path = local_file_path                                                # Создаем папку (открываем)
    video_filename = local_file_path[-31:-4]                                    # Плохое создание оригинального названия

    os.makedirs('frames', exist_ok=True)
    logger.info("📁 Папка 'frames' готова для сохранения кадров")


    old_frames = glob.glob(os.path.join('frames', '*_frame_*.jpg'))             # Удаляем старые кадры
    deleted_count = 0
    for file_path in old_frames:
        try:
            if os.path.isfile(file_path):
                os.remove(file_path)
                deleted_count += 1
        except Exception as e:
            logger.error(f"❌ Ошибка при удалении {file_path}: {e}")

    logger.info(f"🗑️ Удалено {deleted_count} старых кадров из папки 'frames'")


    cap = cv2.VideoCapture(video_path)                                          # Начинаем обработку видео
    if not cap.isOpened():
        logger.error(f"❌ Ошибка: Не могу открыть видеофайл: {video_path}")
        exit()  # завершаем программу если ошибка
    logger.info("✅ Видеофайл успешно открыт!")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    frame_count = 0
    saved_count = 0
    every_n_frame = 30

    progress_bar = tqdm(total=total_frames, desc="Обработка видео")             # Извлекаем кадры

    while True:    
        ret, frame = cap.read()
        if not ret:
            break

        frame_count+=1
        progress_bar.update(1)

        if frame_count%every_n_frame==0:


            filename = f"{video_filename}_frame_{saved_count:04d}.jpg"
            filepath = os.path.join('frames', filename)   
            success = cv2.imwrite(filepath, frame)    

            if success:
                saved_count+=1
            else:
                logger.error(f"❌ Ошибка сохранения кадра {saved_count}")

    progress_bar.close()
    cap.release()

    # ОСВОБОЖДАЕМ ВИДЕОПЛЕЕР (обязательно!)
    cap.release()

    logger.info("✅ Обработка завершена!")
    logger.info(f"📊 Результаты:")
    logger.info(f"   Прочитано кадров: {frame_count}")
    logger.info(f"   Сохранено кадров: {saved_count}")
    logger.info(f"   Кадры сохранены в папку: frames/")

