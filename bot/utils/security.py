import re
import os
import hashlib

def sanitize_filename(filename: str) -> str:
    """
    Обезвреживает имя файла, убирая опасные символы и пути
    """
    # Заменяем опасные символы на подчеркивания
    filename = re.sub(r'[^\w\-_.]', '_', filename)
    
    # Убираем возможные попытки path traversal
    filename = os.path.basename(filename)
    
    # Ограничиваем длину имени
    if len(filename) > 100:
        name, ext = os.path.splitext(filename)
        filename = name[:100-len(ext)] + ext
        
    return filename

