from aiogram import Router, types, F
from aiogram.filters import StateFilter
from aiogram.fsm.context import FSMContext
from keyboards.reply import get_main_reply_keyboard
from states.analysis_states import AnalysisStates
from task_manager import task_manager
from utils.rate_limit import rate_limiter

text_router = Router()

@text_router.message(F.text=="📹 Анализ упражнения")
async def process_analyz(message: types.Message, state: FSMContext):
    if task_manager.has_active_task(message.from_user.id):
        await message.answer(
            "❌ У вас уже есть активная задача обработки видео. "
            "Дождитесь ее завершения или отмените командой /cancel."
        )
        return
    
    await state.set_state(AnalysisStates.waiting_for_video)
    await message.answer(
        "Отправь мне видео с выполнением упражнения.\n"
        "📝 Требования к видео:\n"
        "• Длительность: 5-30 секунд\n"
        "• Хорошее освещение\n"
        "• Вид сбоку или спереди",
        reply_markup=get_main_reply_keyboard()
    )


@text_router.message(F.text=="📚 Инструкция")
async def show_instruction(message: types.Message, state: FSMContext):
    if task_manager.has_active_task(message.from_user.id):
        await message.answer(
            "❌ У вас уже есть активная задача обработки видео. "
            "Дождитесь ее завершения или отмените командой /cancel."
        )
        return
    
    await state.clear()
    await message.answer(
        "📖 Инструкция по использованию бота:\n\n"
        "1. Нажми 'Анализ упражнения'\n"
        "2. Сними или загрузи видео выполнения\n"
        "3. Дождись обработки нейросетью\n"
        "4. Получи детальный анализ техники\n\n"
        "📋 Поддерживаемые упражнения:\n"
        "• Приседания\n• Отжимания\n• Подтягивания\n"
        "• Становая тяга\n• И многие другие!",
        reply_markup=get_main_reply_keyboard()
    )

@text_router.message(F.text=="📊 Мои результаты")
async def show_results(message: types.Message, state: FSMContext):
    if task_manager.has_active_task(message.from_user.id):
        await message.answer(
            "❌ У вас уже есть активная задача обработки видео. "
            "Дождитесь ее завершения или отмените командой /cancel."
        )
        return
    
    await state.clear()
    await message.answer(
        "📈 Здесь будут отображаться твои предыдущие результаты анализа.\n"
        "Функция находится в разработке и будет доступна в ближайшем обновлении!",
        reply_markup=get_main_reply_keyboard()
    )

@text_router.message(~StateFilter(AnalysisStates.waiting_for_video), ~StateFilter(AnalysisStates.processing_video))  
async def handle_other_text(message: types.Message):
    await message.answer(
        "Я понимаю только команды из меню 😊\n"
        "Используй кнопки ниже для навигации:",
        reply_markup=get_main_reply_keyboard()
    )