from aiogram.types import InlineKeyboardButton, InlineKeyboardMarkup

def get_main_inline_keyboard():
    inline_keyboard = InlineKeyboardMarkup(inline_keyboard=[
        [InlineKeyboardButton(text="📹 Анализ упражнения", callback_data="analyz_exercise")],
        [InlineKeyboardButton(text="📚 Инструкция", callback_data="instruction"),
        InlineKeyboardButton(text="📊 Мои результаты", callback_data="my_results")]
    ])
    return inline_keyboard