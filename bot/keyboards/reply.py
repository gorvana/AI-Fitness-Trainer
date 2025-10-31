from aiogram.types import KeyboardButton, ReplyKeyboardMarkup

def get_main_reply_keyboard():
    keyboard = ReplyKeyboardMarkup(
        keyboard=[
            [KeyboardButton(text="📹 Анализ упражнения")],
            [KeyboardButton(text="📚 Инструкция"), 
            KeyboardButton(text="📊 Мои результаты")]],
        resize_keyboard=True,
        one_time_keyboard=False
    )
    return keyboard

