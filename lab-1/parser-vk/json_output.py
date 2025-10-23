# import os
# from dotenv import load_dotenv
# from lib.vk_friends_parser import VKClient, VKFriendsParser

# load_dotenv()

# VK_SERVICE_ACCESS_TOKEN = os.getenv("VK_SERVICE_ACCESS_TOKEN", None)
# if VK_SERVICE_ACCESS_TOKEN is None:
#     raise ValueError(
#     )

# vk = VKClient(token=VK_SERVICE_ACCESS_TOKEN)

# profile_name = ["arsenyc", "a.marchenko3"]
# for i in profile_name:
#     root_id = vk.resolve_screen_name(i)
#     print("ID пользователя:", root_id)

#     parser = VKFriendsParser(
#         vk_client=vk, save_photos=False
#     )
#     parser.fetch_network_fast([root_id], depth=1)

#     parser.save_json(f"user_{root_id}.json")
#     print(f"Данные сохранены в user_{root_id}.json")
import os
import json
from dotenv import load_dotenv
from lib.vk_friends_parser import VKClient, VKFriendsParser

load_dotenv()

VK_SERVICE_ACCESS_TOKEN = os.getenv("VK_SERVICE_ACCESS_TOKEN", None)
if VK_SERVICE_ACCESS_TOKEN is None:
    raise ValueError("VK_SERVICE_ACCESS_TOKEN не найден в переменных окружения")

vk = VKClient(token=VK_SERVICE_ACCESS_TOKEN)

profile_names = ["arsenyc", "a.marchenko3"]

# Инициализируем или загружаем существующие данные
if os.path.exists("nodes.json"):
    with open("nodes.json", 'r', encoding='utf-8') as f:
        try:
            all_data = json.load(f)
        except json.JSONDecodeError:
            # Если файл поврежден, начинаем с пустого списка
            all_data = []
else:
    all_data = []

for profile_name in profile_names:
    root_id = vk.resolve_screen_name(profile_name)
    print("ID пользователя:", root_id)

    parser = VKFriendsParser(
        vk_client=vk, save_photos=False
    )
    parser.fetch_network_fast([root_id], depth=1)

    # Сохраняем во временный файл
    temp_filename = f"user_{root_id}.json"
    parser.save_json(temp_filename)
    print(f"Данные сохранены в {temp_filename}")
    
    # Читаем данные из временного файла
    with open(temp_filename, 'r', encoding='utf-8') as f:
        try:
            user_data = json.load(f)
        except json.JSONDecodeError as e:
            print(f"Ошибка чтения {temp_filename}: {e}")
            continue
    
    # Объединяем данные - приводим к одному типу
    if isinstance(user_data, dict):
        # Если user_data - словарь, добавляем его в список
        if isinstance(all_data, list):
            all_data.append(user_data)
        else:
            # Если all_data не список, создаем новый список
            all_data = [all_data, user_data]
    elif isinstance(user_data, list):
        # Если user_data - список, расширяем им all_data
        if isinstance(all_data, list):
            all_data.extend(user_data)
        else:
            # Если all_data не список, создаем новый список
            all_data = [all_data] + user_data
    
    # Удаляем временный файл (опционально)
    os.remove(temp_filename)
    print(f"Данные пользователя {profile_name} добавлены в общий список")

# Сохраняем объединенные данные
with open("nodes.json", 'w', encoding='utf-8') as f:
    json.dump(all_data, f, ensure_ascii=False, indent=2)

print("Все данные объединены в nodes.json")