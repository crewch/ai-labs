import os
import json
from dotenv import load_dotenv
from lib.vk_friends_parser import VKClient, VKFriendsParser
import random
import pandas as pd
import requests
import time
from datetime import datetime
from collections import defaultdict
import csv
import pandas as pd

# edges = pd.read_csv('edges.csv')
# sampled_edges = edges.drop_duplicates(subset='target_user_id')
# sampled_edges = sampled_edges.sample(n=100)
# set_ids = list(set(sampled_edges['target_user_id']) | set(sampled_edges['interactor_id']))

import os

current_dir_name = os.path.basename(os.getcwd())


profile_names = [260879550]
load_dotenv()

VK_SERVICE_ACCESS_TOKEN = os.getenv("VK_SERVICE_ACCESS_TOKEN", None)
if VK_SERVICE_ACCESS_TOKEN is None:
    raise ValueError("VK_SERVICE_ACCESS_TOKEN не найден в переменных окружения")

vk = VKClient(token=VK_SERVICE_ACCESS_TOKEN)


if os.path.exists("nodes_inference.json"):
    with open("nodes_inference.json", 'r', encoding='utf-8') as f:
        try:
            all_data = json.load(f)
        except json.JSONDecodeError:
            all_data = []
else:
    all_data = []

for profile_name in profile_names:
    root_id = profile_name

    parser = VKFriendsParser(
        vk_client=vk, save_photos=False
    )
    parser.fetch_network_fast([root_id], depth=1)

    temp_filename = f"user_{root_id}.json"
    parser.save_json(temp_filename)
    print(f"Данные сохранены в {temp_filename}")
    
    with open(temp_filename, 'r', encoding='utf-8') as f:
        try:
            user_data = json.load(f)
        except json.JSONDecodeError as e:
            print(f"Ошибка чтения {temp_filename}: {e}")
            continue
    print('++++++++++++++++')


    if (user_data['people']):
        # random_friend = random.choice(user_data['people'])
        # random_friend_id = random.choice(user_data['people'])['id']
        # print(random_friend)
        # print(random_friend_id)
        # parser.fetch_network_fast([random_friend_id], depth=1)

        # # Сохраняем во временный файл
        # temp_filename2 = f"user_{random_friend_id}.json"
        # parser.save_json(temp_filename2)
        # print(f"Данные сохранены в {temp_filename2}")
        
        # Читаем данные из временного файла
        # with open(temp_filename2, 'r', encoding='utf-8') as f:
        #     try:
        #         user_data2 = json.load(f)
        #     except json.JSONDecodeError as e:
        #         print(f"Ошибка чтения {temp_filename2}: {e}")
        #         continue


        if not isinstance(all_data, dict):
            all_data = {"people": [], "edges": []}

        if isinstance(user_data, dict) and "people" in user_data and "edges" in user_data:
            all_data["people"].extend(user_data["people"])
            all_data["edges"].extend(user_data["edges"])
            # all_data["people"].extend(user_data2["people"])
            # all_data["edges"].extend(user_data2["edges"])
        else:
            print(f"Предупреждение: Некорректная структура данных для {profile_name}")
        
        if os.path.exists(temp_filename):
            os.remove(temp_filename)
        # if os.path.exists(temp_filename2):
        #     os.remove(temp_filename2)
        print(f"Данные пользователя {profile_name} добавлены в общий список")

with open("nodes_inference.json", 'w', encoding='utf-8') as f:
    json.dump(all_data, f, ensure_ascii=False, indent=5)

print("Все данные объединены в nodes_inference.json")


# with open('nodes.json', 'r', encoding='utf-8') as f:
#     json_file = json.load(f)

# nodes_ids = pd.json_normalize(json_file['people'])['id'].unique()

# for i in range(len(nodes_ids)):
#     print(i)
#     print(i/len(nodes_ids))
#     node_id = nodes_ids[i]
#     edges_parser(node_id)