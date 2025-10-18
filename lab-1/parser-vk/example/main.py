from dotenv import load_dotenv

load_dotenv()

import os

VK_SERVICE_ACCESS_TOKEN = os.getenv("VK_SERVICE_ACCESS_TOKEN", None)
if VK_SERVICE_ACCESS_TOKEN is None:
    raise ValueError(
        """
        VK_SERVICE_ACCESS_TOKEN is not set.
        Please, create .env file in root directory and add it there.
        More info: https://smmplanner.com/blog/gaid-po-api-vk-kak-podkliuchit-i-ispolzovat
    """
    )

from lib.vk_friends_parser import VKClient, VKFriendsParser

# === 1. Инициализация клиента ===
vk = VKClient(token=VK_SERVICE_ACCESS_TOKEN)

# === 2. Разрешаем короткое имя/ссылку на ID ===
profile_name = "viktor_rudnev"
root_id = vk.resolve_screen_name(profile_name)
print("ID пользователя:", root_id)

# === 3. Собираем друзей (и друзей друзей, если depth=2) ===
parser = VKFriendsParser(
    vk_client=vk, save_photos=False
)  # save_photos сохраняет локально
parser.fetch_network_fast([root_id], depth=2)

# === 4. Строим граф ===
G = parser.build_graph()

print(f"Узлов: {G.number_of_nodes()}, связей: {G.number_of_edges()}")

# === 5. Считаем центральности ===
centr = parser.compute_centralities(G, subject_ids=[root_id])
print("Центральности:", centr)

# === 6. Сохраняем результат ===
parser.save(f"data_{profile_name}.pkl.gz")
print(f"Сохранено в {profile_name}.pkl.gz")

# Просто ID пользователя
USER_ID = root_id

print(f"Собрано людей: {len(parser.people)}")
print(f"Собрано связей: {len(parser.edges)}")

# Строим граф
G = parser.build_graph()
print("Граф построен")

# Считаем центральности (по желанию)
centr = parser.compute_centralities(G, subject_ids=[USER_ID])
print("Центральности для пользователя:", centr[USER_ID])

# Сохраняем результат
parser.save(f"user_{USER_ID}.pkl.gz")
print(f"Данные сохранены в user_{USER_ID}.pkl.gz")

parser.save_json(f"user_{USER_ID}.json")
print(f"Данные сохранены в user_{USER_ID}.json")

# === Визуализация графа с аватарками ===
from pyvis.network import Network

net = Network(
    height="750px", width="100%", bgcolor="#111111", font_color="white", notebook=False
)
net.barnes_hut()

# Добавим узлы вручную с фото
for node_id, node_data in G.nodes(data=True):
    name = (
        f"{node_data.get('first_name', '')} {node_data.get('last_name', '')}".strip()
        or str(node_id)
    )
    photo_url = node_data.get("photo_url") or None

    if photo_url:
        net.add_node(
            node_id,
            title=name,
            shape="image",
            image=photo_url,
            size=250,
            borderWidth=2,
            borderWidthSelected=4,
        )
    else:
        # fallback — без фото
        net.add_node(
            node_id,
            label=name,
            title=name,
            color="#00ccff",
            size=250,
        )

# Добавим рёбра
for source, target in G.edges():
    net.add_edge(source, target)

# Сохраняем граф
html_path = f"vk_graph_{profile_name}_photos.html"
net.save_graph(html_path)

print(f"Граф с аватарками сохранён в {html_path} — открой его в браузере.")
