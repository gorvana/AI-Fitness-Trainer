import glob
import os
import logging
import time
import asyncio
import concurrent.futures
import cv2
from aiogram import Router, F, types
from aiogram.fsm.context import FSMContext
from aiogram.types import FSInputFile
from states.analysis_states import AnalysisStates
from task_manager import task_manager
from utils.rate_limit import rate_limiter
from pose.OpenCV import save_frames
from pose.pose_detection import process_frames_batch, draw_squat_overlay

video_processor_executor = concurrent.futures.ProcessPoolExecutor(max_workers=2)

def get_file_extension(mime_type: str) -> str:
        dict_type = {
            'video/mp4': '.mp4',
            'video/quicktime': '.mov', 
            'video/avi': '.avi',
            'video/x-msvideo': '.avi',
            'video/mpeg': '.mpeg',
            'video/webm': '.webm'
        }
        return dict_type.get(mime_type, '.mp4')


logger = logging.getLogger(__name__)
video_router = Router()

os.makedirs("uploads/videos", exist_ok=True)

@video_router.message(F.video, AnalysisStates.waiting_for_video)
async def handle_exercise_video(message: types.Message, state: FSMContext):
    user_id = message.from_user.id

    if task_manager.has_active_task(user_id):                                           # Проверка на наличие активной задачи
        await message.answer(
            "❌ У вас уже есть активная задача. "
            "Дождитесь завершения или отмените командой /cancel."
        )
        return


    is_limited, remaining, wait_time = await rate_limiter.check_rate_limit(user_id)     # Проверка лимита сообщений
    if is_limited:
        await message.answer(
            "❌ Слишком много запросов!\n"
            f"Доступно запросов: {remaining}/3\n"
            f"Подождите {int(wait_time)} секунд перед отправкой следующего видео."
        )
        return


    try:
        logger.info(f"Получено видео от пользователя {user_id}")            # Валидация видео

        if message.video.duration>60:
            await message.answer(
                "❌ Видео слишком длинное! Пожалуйста, отправьте видео "
                "длительностью до 60 секунд."
            )
            return
        
        if message.video.file_size>(20*1024*1024):
            await message.answer(
                "❌ Файл слишком большой! Максимальный размер - 20MB."
            )
            return

        await state.update_data(                                                        # Засекаем время начала обработки
            processing_start_time=time.time()
        )

        old_frames = glob.glob(os.path.join('uploads/videos', '*'))                 # Удаляем старые видео
        deleted_count = 0
        for file_path in old_frames:
            try:
                if os.path.isfile(file_path):
                    os.remove(file_path)
                    deleted_count += 1
            except Exception as e:
                logger.error(f"❌ Ошибка при удалении {file_path}: {e}")

        timestamp = int(time.time())                                                    # Сохранение видео на компьютер
        user_id = message.from_user.id
        file_extension = get_file_extension(message.video.mime_type)
        filename = f"video_{user_id}_{timestamp}{file_extension}"
        local_file_path = f"uploads/videos/{filename}"     

        await message.answer("💾 Сохраняю видео файл...")
        file_info = await message.bot.get_file(message.video.file_id)
        await message.bot.download_file(file_info.file_path, local_file_path)
        logger.info(f"Файл успешно скачан: {local_file_path}")
        

        await state.set_state(AnalysisStates.processing_video)                          # Анализируем видео 
        await message.answer("🎬 Видео получено! Начинаю анализ...")

        async def process_video_task():                                                 # Создаем функции для асинхронной обработки видео
            try:
                # Запускаем обработку в отдельном процессе
                loop = asyncio.get_event_loop()
                frames = await loop.run_in_executor(
                    video_processor_executor, 
                    save_frames, 
                    local_file_path
                )
                
                if not frames:
                    logger.error("Ошибка при сохранении кадров из видео.")
                    return None

                results = await loop.run_in_executor(
                    video_processor_executor,
                    process_frames_batch,
                    frames
                )

                if not results:
                    logger.error("Ошибка при обработке кадров видео.")
                    return None
                # Найти кадр с минимальным углом колена и подготовить аннотированное изображение
                min_knee_angle = None
                min_result = None
                for r in results:
                    ang = r.get("angles", {})
                    # Берём минимум между левым и правым коленом для данного кадра
                    per_frame_vals = [ang.get("LEFT_KNEE_ANGLE"), ang.get("RIGHT_KNEE_ANGLE")]
                    per_frame_vals = [v for v in per_frame_vals if isinstance(v, (float, float))]
                    if not per_frame_vals:
                        continue
                    local_min = min(per_frame_vals)
                    if min_knee_angle is None or local_min < min_knee_angle:
                        min_knee_angle = local_min
                        min_result = r

                min_knee_frame_path = None
                min_knee_annotated_path = None

                if min_result and isinstance(min_knee_angle, (float)):
                    try:
                        img_path = min_result.get("image_path")
                        if img_path and os.path.isfile(img_path):
                            image = cv2.imread(img_path)
                            if image is not None:
                                # Рисуем линии и подписи углов
                                draw_squat_overlay(
                                    image,
                                    min_result.get("keypoints_pixels", {}),
                                    min_result.get("angles", {})
                                )
                                base = os.path.splitext(os.path.basename(img_path))[0]
                                min_knee_frame_path = img_path
                                min_knee_annotated_path = os.path.join(
                                    "frames", f"{base}_annotated.jpg"
                                )
                                # Сохраняем аннотированный кадр
                                cv2.imwrite(min_knee_annotated_path, image)
                        else:
                            logger.warning("Путь к изображению минимального угла некорректен или файл не найден.")
                    except Exception as e:
                        logger.error(f"Ошибка при подготовке аннотированного кадра: {e}")
                        min_knee_annotated_path = None
                summary = {
                    "frames_count": len(frames),
                    "processed_count": len(results),
                    "min_knee_angle": min_knee_angle,
                    "min_knee_frame_path": min_knee_frame_path,
                    "min_knee_annotated_path": min_knee_annotated_path,
                }
                return summary
                
            except Exception as e:
                logger.error(f"Ошибка в задаче обработки: {e}")
                return False

        
        video_task = asyncio.create_task(process_video_task())                          # Создаем и регистрируем задачу
        task_manager.register_task(user_id, video_task)

        
        try:                                                                            # Ждем завершения задачи (с возможностью отмены)
            summary = await video_task
            # Если задача завершилась (даже с ошибкой)

            if summary:
                frames_count = summary["frames_count"]
                processed_count = summary["processed_count"]
                min_knee = summary["min_knee_angle"]
                text_min_knee = f"{int(min_knee)}°" if isinstance(min_knee, (int, float)) else "—"
                
                user_data = await state.get_data()

                await message.answer(
                    (
                        f"✅ Видео обработано за {time.time()-user_data.get('processing_start_time', 0):.2f} секунд!\n"
                        f"📸 Кадров сохранено: {frames_count}\n"
                        f"🧠 Кадров проанализировано: {processed_count}\n"
                        f"🦵 Минимальный угол в колене: {text_min_knee}\n\n"
                        "Совет: старайтесь держать корпус стабильно и колени направлять по носкам."
                    )
                )

                # Отправляем пользователю кадр с минимальным углом колена (с разметкой)
                annotated_path = summary.get("min_knee_annotated_path")
                if annotated_path and os.path.isfile(annotated_path):
                    try:
                        photo = FSInputFile(annotated_path)
                        await message.answer_photo(
                            photo=photo,
                            caption=f"Кадр с минимальным углом колена: {text_min_knee}"
                        )
                    except Exception as e:
                        logger.error(f"Не удалось отправить аннотированный кадр: {e}")
            else:
                await message.answer("❌ Ошибка при обработке видео")
                
        except asyncio.CancelledError:
            # Сюда попадем, если задачу отменили через task_manager                 
            await message.answer("Запрос на отмену обработки принят")
            return
            
        finally:
            # ВСЕГДА убираем задачу из менеджера при завершении
            task_manager.remove_completed_task(user_id)

        await state.clear()

    except Exception as e:
        logger.error(f"Ошибка при обработке видео: {e}")
        await message.answer("❌ Произошла ошибка при обработке видео. Попробуйте еще раз.")
        await state.clear() 



@video_router.message(AnalysisStates.waiting_for_video)
async def handle_wrong_content_type(message: types.Message):
    await message.answer(
        "❌ Пожалуйста, отправьте именно видео файл с упражнением.\n"
        "Вы можете записать видео прямо в Telegram или выбрать из галереи.\n\n"
        "Если вы передумали, отправьте /cancel для отмены."
    )


@video_router.message(AnalysisStates.processing_video)
async def analys_video(message: types.Message):
    await message.answer(
        "❌ Пожалуйста, подождите пока видео обработается.\n\n"
        "Если вы передумали, отправьте /cancel для отмены."
    )


async def process_video_async(file_path: str):
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, save_frames, file_path)
