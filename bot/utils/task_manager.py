import asyncio
from typing import Dict
import logging

logger = logging.getLogger(__name__)

class TaskManager:
    def __init__(self):
        self.active_tasks: Dict[int, asyncio.Task] = {}
    
    def register_task(self, user_id: int, task: asyncio.Task):
        self.active_tasks[user_id] = task
        logger.info(f"📝 Зарегистрирована задача для пользователя {user_id}")
    
    async def cancel_user_task(self, user_id: int) -> bool:
        if user_id in self.active_tasks:
            task = self.active_tasks[user_id]
            if not task.done():  
                task.cancel()    
                try:
                    await task  
                except asyncio.CancelledError:
                    logger.info(f"✅ Задача пользователя {user_id} успешно отменена")
                finally:
                    self.active_tasks.pop(user_id, None)
                    return True
            else:
                self.active_tasks.pop(user_id, None)
        return False
    
    def remove_completed_task(self, user_id: int):
        if user_id in self.active_tasks and self.active_tasks[user_id].done():
            self.active_tasks.pop(user_id, None)
            logger.info(f"🗑️ Удалена завершенная задача пользователя {user_id}")

    def has_active_task(self, user_id: int) -> bool:
        return user_id in self.active_tasks and not self.active_tasks[user_id].done()


task_manager = TaskManager() 