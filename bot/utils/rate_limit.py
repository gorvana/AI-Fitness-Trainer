import time
from typing import Dict, List, Tuple
import logging

logger = logging.getLogger(__name__)

class RateLimiter:
    def __init__(self):
        self.user_requests: Dict[int, List[float]] = {}
    
    async def check_rate_limit(self, user_id: int, max_requests: int = 3, time_window: int = 60) -> Tuple[bool, int, float]:

        current_time = time.time()
        
        if user_id not in self.user_requests:
            self.user_requests[user_id] = []
        
        recent_requests = [
            req_time for req_time in self.user_requests[user_id] 
            if current_time - req_time < time_window
        ]
        
        is_limited = True if len(recent_requests) >= max_requests else False
        remaining_requests = max(0, max_requests - len(recent_requests))
        
        wait_time = 0.0
        if is_limited and recent_requests:
            oldest_request = min(recent_requests)
            wait_time = max(0, time_window - (current_time - oldest_request))
        else:
            wait_time = 0.0

        return is_limited, remaining_requests, wait_time
        
    async def add_request(self, user_id: int):
        try:
            self.user_requests[user_id].append(time.time())
            logger.info(f"✅ Запрос пользователя {user_id} добавлен. Всего: {len(self.user_requests[user_id])}/3")
        except Exception as e:
            logger.warning(f"❌ Ошибка при добавлении запроса для пользователя {user_id}: {e}")
        
        


rate_limiter = RateLimiter()