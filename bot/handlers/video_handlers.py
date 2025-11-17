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
from utils.task_manager import task_manager
from utils.rate_limit import rate_limiter
from utils.OpenCV import save_frames
from utils.pose_detection import process_frames_batch, draw_squat_overlay
from utils.dataset_writer import write_sequence_record, ERROR_LABELS_ORDER
from utils.dataset_writer import _canonicalize_error_label

video_processor_executor = concurrent.futures.ProcessPoolExecutor(max_workers=2)
logger = logging.getLogger(__name__)
video_router = Router()


@video_router.message(F.video, AnalysisStates.waiting_for_video)
async def handle_exercise_video(message: types.Message, state: FSMContext):
    def get_file_extension(mime_type: str) -> str:                                              # Создание функции для получение расширения файла по MIME типу                                
        dict_type = {
            'video/mp4': '.mp4',
            'video/quicktime': '.mov', 
            'video/avi': '.avi',
            'video/x-msvideo': '.avi',
            'video/mpeg': '.mpeg',
            'video/webm': '.webm'
        }
        if mime_type not in dict_type:
            raise ValueError(f"Unsupported MIME type: {mime_type}")
        return dict_type[mime_type]

    async def validate_video(user_id: int) -> bool:                                             # Создание функции для валидации видео 
            if message.video.duration > 60:                                                         # Максимальная длительность 60 секунд                 
                await message.answer(
                "❌ Видео слишком длинное! Пожалуйста, отправьте видео "
                "длительностью до 60 секунд."
                )
                return False
            if message.video.file_size > (20 * 1024 * 1024):                                        # Максимальный размер файла 20MB  
                await message.answer(
                "❌ Файл слишком большой! Максимальный размер - 20MB."
                )
                return False
            if task_manager.has_active_task(user_id):                                               # Проверка на наличие активной задачи
                await message.answer(
                "❌ У вас уже есть активная задача. "
                "Дождитесь завершения или отмените командой /cancel."
                )
                return False
            is_limited, remaining_requests, wait_time = await rate_limiter.check_rate_limit(user_id)  # Проверка лимита сообщений
            if is_limited:
                await message.answer(
                "❌ Слишком много запросов!\n"
                f"Доступно запросов: {remaining_requests}/3\n"
                f"Дождитесь завершения {int(wait_time)} секунд или отмените командой /cancel."
                )
                return False
            else:
                await rate_limiter.add_request(user_id)
            return True
    
    async def save_video_file():                                                                # Создание функции для сохранения видео файла
            try:
                os.makedirs("uploads/videos", exist_ok=True)                                        # Создаем папку для загрузки видео, если её нет

                timestamp = int(time.time())                                                        # Создаём уникальное имя и сохраняем видео на компьютер
                file_extension = get_file_extension(message.video.mime_type)
                filename = f"video_{message.from_user.id}_{timestamp}{file_extension}"
                local_file_path = f"uploads/videos/{filename}"

                await message.answer("💾 Сохраняю видео файл...")                                   # Сохраняем файл
                file_info = await message.bot.get_file(message.video.file_id)
                await message.bot.download_file(file_info.file_path, local_file_path)
                logger.info(f"Файл успешно скачан: {local_file_path}")
                return True, local_file_path
            
            except Exception as e:
                logger.error(f"Ошибка при сохранении видео файла: {e}")
                await message.answer("❌ Ошибка при сохранении видео файла.")
                return False, None

    async def process_video_task(video_path: str):                                              # Создаем функцию для асинхронной обработки видео
        try:

            loop = asyncio.get_event_loop()                                                 
            frames_path = await loop.run_in_executor(                                               # Запускаем функцию для сохранения кадров и получения путей в отдельном процессе
                video_processor_executor, 
                save_frames, 
                video_path
            )
            if not frames_path:
                logger.error("Ошибка при сохранении кадров из видео.")
                return None


            results = await loop.run_in_executor(                                                   # Запускаем функцию для обработки кадров и получения словарей в отдельном процессе
                video_processor_executor,
                process_frames_batch,
                frames_path
            )
            if not results:
                logger.error("Ошибка при обработке кадров видео.")
                return None
                    

            min_knee_angle = None                                                                   # Находим минимальный угол колена
            min_result = None
            min_knee_annotated_path = None

            for res in results:                                                                         
                ang = res.get("angles", {})
                per_frame_vals = [ang.get("LEFT_KNEE_ANGLE"), ang.get("RIGHT_KNEE_ANGLE")]
                per_frame_vals = [v for v in per_frame_vals if isinstance(v, (float, float))]
                if not per_frame_vals:
                    continue
                if min_knee_angle is None or min(per_frame_vals) < min_knee_angle:
                    min_knee_angle = min(per_frame_vals)
                    min_result = res
                    min_knee_annotated_path = res["image_path"]

                                                                    
            
            if min_result and isinstance(min_knee_angle, (float)):                                  # Готовим аннотированный кадр с минимальным углом колена    
                try:
                    if min_knee_annotated_path and os.path.isfile(min_knee_annotated_path):
                        image = cv2.imread(min_knee_annotated_path)
                        if image is not None:                                    
                            draw_squat_overlay(                                                     # Рисуем линии и подписи углов
                                image,
                                min_result.get("keypoints_pixels", {}),
                                min_result.get("angles", {})
                            )
                            base = os.path.splitext(os.path.basename(min_knee_annotated_path))[0]
                            min_knee_annotated_path = os.path.join(
                                "frames", f"{base}_annotated.jpg"
                            )
                            cv2.imwrite(min_knee_annotated_path, image)
                    else:
                        logger.warning("Путь к изображению минимального угла некорректен или файл не найден.")
                except Exception as e:
                    logger.error(f"Ошибка при подготовке аннотированного кадра: {e}")
                    
                    
            summary = {                                                                             # Готовим итоговый словарь с результатами              
                "processed_count": len(results),
                "min_knee_angle": min_knee_angle,
                "min_knee_annotated_path": min_knee_annotated_path,
                "results": results
            }
            return summary
                    
        except Exception as e:
            logger.error(f"Ошибка в задаче обработки: {e}")
            return False    

    try:

        validate_video = await validate_video(message.from_user.id)                             # Валидация видео
        if not validate_video:
            return
        
        success, video_path = await save_video_file()                                           # Сохранение видео файла
        if not success:
            return

        caption = (message.caption or "").strip() if hasattr(message, "caption") else ""        # Захватываем метку ошибки из подписи к видео (caption). Это позволит вручную размечать датасет
        await state.update_data(                                                            
            processing_start_time=time.time(),                                                  # Засекаем время начала обработки и сохраняем метку
            error_label=caption                                                                
        )
    
        if caption:                                                                             # Отправляем сообщение о начале обработки видео
            await message.answer(
                "🎬 Видео получено! Начинаю анализ...\n"
                f"🏷️ Метка ошибки (из подписи): {caption}"
            )
        else:
            await message.answer(
                "🎬 Видео получено! Начинаю анализ...\n"
                "ℹ️ Для ручной разметки добавляйте в подпись к видео тип ошибки: "
                "knees_in | shallow_depth | heels_off | forward_lean (можно на русском)"
            )

        await state.set_state(AnalysisStates.processing_video)                                  # Меняем состояние на обработку видео




        video_task = asyncio.create_task(process_video_task(video_path))                        # Создаем и регистрируем задачу
        task_manager.register_task(message.from_user.id, video_task)

        
        try:                                                                                    # Ждем завершения задачи (с возможностью отмены)
            summary = await video_task
            # Если задача завершилась (даже с ошибкой)

            if summary:                                                                         # Отправляем результаты обработки пользователю
                processed_count = summary["processed_count"]
                min_knee = summary["min_knee_angle"]
                text_min_knee = f"{int(min_knee)}°" if isinstance(min_knee, (int, float)) else "—"
                
                user_data = await state.get_data()

                await message.answer(
                    (
                        f"✅ Видео обработано за {time.time()-user_data.get('processing_start_time', 0):.2f} секунд!\n"
                        f"🧠 Кадров проанализировано: {processed_count}\n"
                        f"🦵 Минимальный угол в колене: {text_min_knee}\n\n"
                        "Совет: старайтесь держать корпус стабильно и колени направлять по носкам."
                    )
                )

                                                                                        
                annotated_path = summary.get("min_knee_annotated_path")                         # Отправляем аннотированный кадр с минимальным углом колена
                if annotated_path and os.path.isfile(annotated_path):
                    try:
                        photo = FSInputFile(annotated_path)
                        await message.answer_photo(
                            photo=photo,
                            caption=f"Кадр с минимальным углом колена: {text_min_knee}"
                        )
                    except Exception as e:
                        logger.error(f"Не удалось отправить аннотированный кадр: {e}")



                
                try:                                                                        # Запись обучающей выборки (Keras-ready)
                    err_label = (user_data.get("error_label") or "").strip()
                    canon = _canonicalize_error_label(err_label)
                    if canon is None:
                        logger.error(f"Некорректная метка ошибки: {err_label}")
                        await message.answer("⚠️ Некорректная метка ошибки для разметки датасета. Пример корректных меток: knees_in, shallow_depth, heels_off, forward_lean. Для нескольких меток используйте разделитель запятая/точка с запятой/|, например: 'knees_in, forward_lean'.")
                    else:
                        active = [k for k, v in canon.get("labels", {}).items() if v]
                        labels_display = ", ".join(active) if active else (canon.get("label") or "—")
                        await message.answer(
                            f"🏷️ Метки для записи: {labels_display}. Записываю обучающий пример в датасет..."
                        )
                        seq_path = write_sequence_record(summary, video_path, err_label)
                        await message.answer(
                            "📦 Обучающий пример добавлен в датасет.\n"
                            f"Файл последовательностей: {seq_path}"
                        )
                except Exception as e:
                    logger.error(f"Ошибка записи датасета: {e}")
                    await message.answer("⚠️ Не удалось сохранить данные для обучения.")
            else:
                await message.answer("❌ Ошибка при обработке видео")
                
        except asyncio.CancelledError:               
            await message.answer("Запрос на отмену обработки принят")
            return
            
        finally:
            task_manager.remove_completed_task(message.from_user.id)

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
