from Token import BOT_TOKEN
from aiogram import Bot, Dispatcher
import asyncio
import logging


from handlers.callback_handlers import callback_router
from handlers.text_handlers import text_router
from handlers.user_commands import user_commands_router
from handlers.video_handlers import video_router

logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'  
)

logger = logging.getLogger(__name__)

bot = Bot(token=BOT_TOKEN)
dp = Dispatcher()

dp.include_router(user_commands_router)
dp.include_router(text_router)
dp.include_router(callback_router)
dp.include_router(video_router) 

async def main():
    print("Бот запущен!")
    await dp.start_polling(bot)

if __name__ == "__main__":
    asyncio.run(main())