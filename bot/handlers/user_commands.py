from aiogram import Router, types
from aiogram.filters import Command
from keyboards.inline import get_main_inline_keyboard
from aiogram.fsm.context import FSMContext
from states.analysis_states import AnalysisStates
from task_manager import task_manager

user_commands_router = Router()

@user_commands_router.message(Command("start"))
async def cmd_start(message: types.Message, state: FSMContext):
    await state.clear()
    user_name = message.from_user.first_name

    await message.answer(
        f"Привет, {user_name}! 🏋️\n"
        "Я бот для анализа правильности выполнения упражнений.\n"
        "Выбери действие:",
        reply_markup=get_main_inline_keyboard()   
    )


@user_commands_router.message(Command("cancel"))
async def cmd_cancel(message: types.Message, state: FSMContext):
    user_id = message.from_user.id
    current_state = await state.get_state()

    if current_state == AnalysisStates.processing_video.state:
        
        success = await task_manager.cancel_user_task(user_id)
        
        if success:
            await state.clear()
            await message.answer("✅ Обработка видео успешно отменена!")
        else:
            await message.answer("⚠️ Не удалось отменить обработку. Возможно, она уже завершена.")
            
    elif current_state == AnalysisStates.waiting_for_video.state:
        await state.clear()
        await message.answer("✅ Ожидание видео отменено!")
    else:
        await message.answer("❌ Нет активных задач для отмены")