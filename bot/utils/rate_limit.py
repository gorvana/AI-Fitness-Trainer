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
        
        user_requests = self.user_requests[user_id]
        recent_requests = [
            req_time for req_time in user_requests 
            if current_time - req_time < time_window
        ]
        
        requests_count = len(recent_requests)
        is_limited = requests_count >= max_requests
        remaining_requests = max(0, max_requests - requests_count)
        
        wait_time = 0.0
        if is_limited and recent_requests:
            oldest_request = min(recent_requests)
            wait_time = max(0, time_window - (current_time - oldest_request))
        
        if not is_limited:
            recent_requests.append(current_time)
            self.user_requests[user_id] = recent_requests
            logger.info(f"✅ Запрос пользователя {user_id} добавлен. Всего: {len(recent_requests)}/{max_requests}")
        else:
            logger.warning(f"❌ Пользователь {user_id} превысил лимит: {requests_count}/{max_requests}")
        
        return is_limited, remaining_requests, wait_time


rate_limiter = RateLimiter()