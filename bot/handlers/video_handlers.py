from aiogram import Router, F, types
from aiogram.fsm.context import FSMContext
from states.analysis_states import AnalysisStates
import logging
import time
import os
from OpenCV import save_frames
import asyncio
import concurrent.futures
from task_manager import task_manager
from utils.rate_limit import rate_limiter
from utils.security import sanitize_filename

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
        logger.info(f"Получено видео от пользователя {user_id}")                         # Валидация видео

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

        timestamp = int(time.time())
        file_extension = get_file_extension(message.video.mime_type)
        original_filename = sanitize_filename(f"video_{user_id}_{timestamp}{file_extension}")
        local_file_path = f"uploads/videos/{original_filename}"    

        await message.answer("💾 Сохраняю видео файл...")
        file_info = await message.bot.get_file(message.video.file_id)
        await message.bot.download_file(file_info.file_path, local_file_path)
        logger.info(f"Файл успешно скачан: {local_file_path}")

        await asyncio.sleep(5)
        

        await state.set_state(AnalysisStates.processing_video)                          # Анализируем видео 
        await message.answer("🎬 Видео получено! Начинаю анализ...")

        async def process_video_task():                                                 # Создаем функции для асинхронной обработки видео
            try:
                # Запускаем обработку в отдельном процессе
                loop = asyncio.get_event_loop()
                await loop.run_in_executor(
                    video_processor_executor, 
                    save_frames, 
                    local_file_path
                )
                
                # Если дошли сюда - обработка завершена успешно
                return True
            except Exception as e:
                logger.error(f"Ошибка в задаче обработки: {e}")
                return False

        
        video_task = asyncio.create_task(process_video_task())                          # Создаем и регистрируем задачу
        task_manager.register_task(user_id, video_task)

        
        try:                                                                            # Ждем завершения задачи (с возможностью отмены)
            success = await video_task
            # Если задача завершилась (даже с ошибкой)

            if success:
                analysis_result = {
                    'technique_score': 85,
                    'amplitude': 'Хорошо',
                    'speed': 'Нормальная', 
                    'recommendation': 'Следите за положением спины'
                }
                
                await message.answer(
                    f"✅ Видео обработано!\n"
                    f"📊 Результаты:\n"
                    f"• Техника выполнения: {analysis_result['technique_score']}%\n"
                    f"• Амплитуда движения: {analysis_result['amplitude']}\n"
                    f"• Скорость выполнения: {analysis_result['speed']}\n" 
                    f"• Рекомендации: {analysis_result['recommendation']}"
                )
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
