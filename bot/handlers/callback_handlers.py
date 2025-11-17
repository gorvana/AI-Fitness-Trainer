from aiogram import Router, types, F
from keyboards.reply import get_main_reply_keyboard
from aiogram.fsm.context import FSMContext
from states.analysis_states import AnalysisStates
from utils.task_manager import task_manager

callback_router = Router()

@callback_router.callback_query(F.data=="analyz_exercise")
async def process_analyz(callback_query: types.CallbackQuery, state: FSMContext):
    await callback_query.answer()
    if task_manager.has_active_task(callback_query.from_user.id):
        await callback_query.message.answer(
            "❌ У вас уже есть активная задача обработки видео. "
            "Дождитесь ее завершения или отмените командой /cancel."
        )
        return

    
    await state.set_state(AnalysisStates.waiting_for_video)
    await callback_query.message.answer(
        "Отправь мне видео с выполнением упражнения.\n"
        "📝 Требования к видео:\n"
        "• Длительность: 5-30 секунд\n"
        "• Хорошее освещение\n"
        "• Вид сбоку или спереди",
        reply_markup=get_main_reply_keyboard()
    )


@callback_router.callback_query(F.data=="instruction")
async def show_instruction(callback_query: types.CallbackQuery, state: FSMContext):
    await callback_query.answer()
    if task_manager.has_active_task(callback_query.from_user.id):
        await callback_query.message.answer(
            "❌ У вас уже есть активная задача обработки видео. "
            "Дождитесь ее завершения или отмените командой /cancel."
        )
        return
    
    await state.clear()
    await callback_query.message.answer(
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


@callback_router.callback_query(F.data=="my_results")
async def show_results(callback_query: types.CallbackQuery, state: FSMContext):
    await callback_query.answer()
    if task_manager.has_active_task(callback_query.from_user.id):
        await callback_query.message.answer(
            "❌ У вас уже есть активная задача обработки видео. "
            "Дождитесь ее завершения или отмените командой /cancel."
        )
        return
    
    await state.clear()
    await callback_query.message.answer(
        "📈 Здесь будут отображаться твои предыдущие результаты анализа.\n"
        "Функция находится в разработке и будет доступна в ближайшем обновлении!",
        reply_markup=get_main_reply_keyboard()
    )