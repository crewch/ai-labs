import requests
import json
import time
from datetime import datetime
from collections import defaultdict
from datetime import datetime
import csv
import pandas as pd

class VKParser:
    def __init__(self, access_token, version='5.131'):
        self.access_token = access_token
        self.version = version
        self.base_url = 'https://api.vk.com/method/'
        
    def make_request(self, method, params):
        """Базовый метод для запросов к VK API"""
        url = f"{self.base_url}{method}"
        params.update({
            'access_token': self.access_token,
            'v': self.version
        })
        
        try:
            response = requests.get(url, params=params)
            data = response.json()
            
            if 'error' in data:
                print(f"Ошибка VK API: {data['error']['error_msg']}")
                return None
                
            return data['response']
        except Exception as e:
            print(f"Ошибка запроса: {e}")
            return None

    def resolve_screen_name(self, screen_name):
        print(f"Преобразуем короткое имя '{screen_name}' в ID...")
        
        response = self.make_request('utils.resolveScreenName', {
            'screen_name': screen_name
        })
        
        if response and 'object_id' in response:
            user_id = response['object_id']
            print(f"Найден ID: {user_id}")
            return user_id
        else:
            print(f"Не удалось найти ID для '{screen_name}'")
            return None

    def get_user_id(self, user_input):

        if isinstance(user_input, int) or (isinstance(user_input, str) and user_input.isdigit()):
            return int(user_input)
        
        # Если это короткое имя, преобразуем в ID
        if isinstance(user_input, str) and not user_input.startswith('-'):
            return self.resolve_screen_name(user_input)
        
        return user_input

    def get_wall_posts(self, owner_id, count=100):
        """Получает посты со стены пользователя"""
        print(f"Получаем посты со стены пользователя {owner_id}...")
        
        # Преобразуем owner_id в числовой формат если нужно
        numeric_owner_id = self.get_user_id(owner_id)
        if numeric_owner_id is None:
            print(f"Не удалось определить ID для {owner_id}")
            return []
        
        posts = []
        offset = 0
        max_posts = count
        
        while len(posts) < max_posts:
            response = self.make_request('wall.get', {
                'owner_id': numeric_owner_id,
                'count': min(100, max_posts - len(posts)),
                'offset': offset,
                'extended': 1
            })
            
            if not response or 'items' not in response:
                break
                
            posts.extend(response['items'])
            offset += len(response['items'])
            
            if len(response['items']) == 0:
                break
                
            time.sleep(0.34)
        
        print(f"Получено {len(posts)} постов")

        posts_count = pd.read_csv('posts_count.csv')

        new_row = pd.DataFrame({
            'id': [numeric_owner_id],
            'posts': [len(posts)]
        })

        posts_count = pd.concat([posts_count, new_row], ignore_index=True)
        posts_count.to_csv('posts_count.csv', index=False)



        return posts, numeric_owner_id
    
    def get_likes(self, owner_id, item_id, item_type='post'):
        """Получает список пользователей, лайкнувших запись"""
        likes = []
        offset = 0
        count = 1000
        
        while True:
            response = self.make_request('likes.getList', {
                'type': item_type,
                'owner_id': owner_id,
                'item_id': item_id,
                'count': count,
                'offset': offset,
                'filter': 'likes'
            })
            
            if not response or 'items' not in response:
                break
                
            likes.extend(response['items'])
            offset += len(response['items'])
            
            if len(response['items']) < count:
                break
                
            time.sleep(0.34)
        
        return likes
    
    def get_comments(self, owner_id, post_id):
        """Получает комментарии к посту"""
        comments = []
        offset = 0
        count = 100
        
        while True:
            response = self.make_request('wall.getComments', {
                'owner_id': owner_id,
                'post_id': post_id,
                'count': count,
                'offset': offset,
                'extended': 0
            })
            
            if not response or 'items' not in response:
                break
                
            for comment in response['items']:
                comment_data = {
                    'id': comment['id'],
                    'from_id': comment['from_id'],
                    'date': comment['date'],
                    'text': comment['text'],
                    'likes': comment.get('likes', {}).get('count', 0)
                }
                comments.append(comment_data)
            
            offset += len(response['items'])
            
            if len(response['items']) < count:
                break
                
            time.sleep(0.34)
        
        return comments
    
    def get_reposts(self, owner_id, post_id):
        """Получает информацию о репостах"""
        response = self.make_request('wall.getReposts', {
            'owner_id': owner_id,
            'post_id': post_id,
            'count': 1000
        })
        
        if not response:
            return []
            
        reposts = []
        if 'items' in response:
            for repost in response['items']:
                repost_data = {
                    'id': repost['id'],
                    'from_id': repost['from_id'],
                    'date': repost['date'],
                    'text': repost.get('text', ''),
                    'copy_history': repost.get('copy_history', [])
                }
                reposts.append(repost_data)
        
        return reposts
    
    def get_user_info(self, user_ids):
        """Получает информацию о пользователях"""
        if not user_ids:
            return {}
            
        numeric_ids = []
        for user_id in user_ids:
            if isinstance(user_id, int) or (isinstance(user_id, str) and user_id.lstrip('-').isdigit()):
                numeric_ids.append(str(user_id))
        
        if not numeric_ids:
            return {}
            
        response = self.make_request('users.get', {
            'user_ids': ','.join(numeric_ids),
            'fields': 'first_name,last_name,sex,bdate,city,country,screen_name'
        })
        
        if not response:
            return {}
            
        user_info = {}
        for user in response:
            user_info[user['id']] = {
                'first_name': user.get('first_name', ''),
                'last_name': user.get('last_name', ''),
                'sex': user.get('sex', 0),
                'bdate': user.get('bdate', ''),
                'city': user.get('city', {}).get('title', '') if 'city' in user else '',
                'country': user.get('country', {}).get('title', '') if 'country' in user else '',
                'screen_name': user.get('screen_name', '')
            }
        
        return user_info
    
    def process_post(self, post, owner_id):
        """Обрабатывает один пост и собирает всю информацию"""
        post_id = post['id']
        
        print(f"Обрабатываем пост {post_id}...")
        
        post_data = {
            'post_id': post_id,
            'owner_id': owner_id,
            'date': post['date'],
            'text': post.get('text', '')[:500],
            'likes_count': post.get('likes', {}).get('count', 0),
            'comments_count': post.get('comments', {}).get('count', 0),
            'reposts_count': post.get('reposts', {}).get('count', 0),
            'views_count': post.get('views', {}).get('count', 0) if 'views' in post else 0
        }
        
        if 'copy_history' in post and post['copy_history']:
            original_post = post['copy_history'][0]
            post_data['is_repost'] = True
            post_data['reposted_from'] = {
                'owner_id': original_post['owner_id'],
                'post_id': original_post['id'],
                'text': original_post.get('text', '')[:500]
            }
        else:
            post_data['is_repost'] = False
            post_data['reposted_from'] = None
        
        if isinstance(owner_id, int):
            print(f"  Собираем лайки...")
            post_data['likes'] = self.get_likes(owner_id, post_id)
            time.sleep(0.34)
            
            print(f"  Собираем комментарии...")
            post_data['comments'] = self.get_comments(owner_id, post_id)
            time.sleep(0.34)
            
            print(f"  Собираем репосты...")
            post_data['reposts'] = self.get_reposts(owner_id, post_id)
            time.sleep(0.34)
        else:
            print(f"  Пропускаем сбор лайков/комментариев - неверный owner_id")
            post_data['likes'] = []
            post_data['comments'] = []
            post_data['reposts'] = []
        
        return post_data
    
    def parse_user_wall(self, user_input, max_posts=50):
        print(f"\n=== Начинаем парсинг стены пользователя {user_input} ===")
        
        posts, numeric_owner_id = self.get_wall_posts(user_input, max_posts)
        processed_posts = []
        
        for i, post in enumerate(posts):
            print(f"Пост {i+1}/{len(posts)}")
            processed_post = self.process_post(post, numeric_owner_id)
            processed_posts.append(processed_post)
            
            if i < len(posts) - 1:
                time.sleep(1)
        
        return processed_posts, numeric_owner_id
    
    def collect_all_user_ids(self, posts_data):
        user_ids = set()
        
        for post in posts_data:
            user_ids.update(post['likes'])
            
            for comment in post['comments']:
                user_ids.add(comment['from_id'])
            
            for repost in post['reposts']:
                user_ids.add(repost['from_id'])
        
        return list(user_ids)
    
    def analyze_two_users(self, user1_input, user2_input, max_posts_per_user=20):
        print(f"Запускаем анализ пользователей {user1_input} и {user2_input}")
        
        user1_posts, user1_id = self.parse_user_wall(user1_input, max_posts_per_user)
        user2_posts, user2_id = self.parse_user_wall(user2_input, max_posts_per_user)
        
        all_user_ids = set()
        all_user_ids.update(self.collect_all_user_ids(user1_posts))
        all_user_ids.update(self.collect_all_user_ids(user2_posts))
        
        print(f"Собираем информацию о {len(all_user_ids)} пользователях...")
        user_info = self.get_user_info(list(all_user_ids))
        
        result = {
            'analysis_info': {
                'timestamp': datetime.now().isoformat(),
                'user1_input': user1_input,
                'user2_input': user2_input,
                'user1_id': user1_id,
                'user2_id': user2_id,
                'total_posts_analyzed': len(user1_posts) + len(user2_posts),
                'total_users_found': len(all_user_ids)
            },
            'user1_posts': user1_posts,
            'user2_posts': user2_posts,
            'user_info': user_info
        }
        
        return result
    
    def save_to_json(self, data, filename=None):
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            user1 = str(data['analysis_info']['user1_input']).replace('/', '_')
            user2 = str(data['analysis_info']['user2_input']).replace('/', '_')
            filename = f"vk_analysis_{user1}_{user2}_{timestamp}.json"
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        print(f"Данные сохранены в файл: {filename}")
        return filename
    
    def generate_interaction_csv(self, user_input, output_filename=None, max_posts=100):
        print(f"Генерируем CSV со статистикой взаимодействий для {user_input}...")
        

        posts, owner_id = self.parse_user_wall(user_input, max_posts)
        
        if not posts:
            print("Не удалось получить посты пользователя")
            return None
        
        interactions = defaultdict(lambda: {'likes': 0, 'comments': 0, 'reposts': 0})
        
        for post in posts:
            self._process_post_for_interactions(post, interactions, owner_id)
        
        if not output_filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            user_identifier = str(user_input).replace('/', '_')
            output_filename = f"vk_interactions_{user_identifier}_{timestamp}.csv"
        
        return self._write_interactions_to_csv(owner_id, interactions, output_filename)
    
    def _process_post_for_interactions(self, post, interactions, owner_id):
        post_id = post['post_id']
        
        for like_user_id in post.get('likes', []):
            interactions[like_user_id]['likes'] += 1
        
        for comment in post.get('comments', []):
            commenter_id = comment['from_id']
            interactions[commenter_id]['comments'] += 1
        
        for repost in post.get('reposts', []):
            reposter_id = repost['from_id']
            interactions[reposter_id]['reposts'] += 1
    
    def _write_interactions_to_csv(self, owner_id, interactions, output_filename):
        target_file = 'edges.csv'

        try:
            edges = pd.read_csv(target_file)
        except FileNotFoundError:
            edges = pd.DataFrame(columns=[
                'target_user_id',
                'interactor_id',
                'likes_count',
                'comments_count',
                'reposts_count'
            ])


        new_rows = []
        for interactor_id, stats in interactions.items():
            new_rows.append({
                'target_user_id': owner_id,
                'interactor_id': interactor_id,
                'likes_count': stats['likes'],
                'comments_count': stats['comments'],
                'reposts_count': stats['reposts']
            })

        if new_rows:
            new_df = pd.DataFrame(new_rows)
            edges = pd.concat([edges, new_df], ignore_index=True)

        print(edges)
        edges.to_csv(target_file, index=False)

        print(f"CSV файл обновлён: {target_file}")
        print(f"Добавлено взаимодействий: {len(new_rows)}")
        return target_file

def edges_parser(wall_id):
    # Настройки
    VK_ACCESS_TOKEN = 'e68624dce68624dce68624dc9ee5bde388ee686e68624dc8e7589f8b43544c74ac5ca7c'  # Замените на ваш токен
    USER1_ID = 'a.marchenko3'  # Можно использовать короткие имена
    USER2_ID = 'arsenyc'      # Можно использовать короткие имена
    MAX_POSTS_PER_USER = 10
    
    # Создаем парсер
    parser = VKParser(VK_ACCESS_TOKEN)

    csv_file = parser.generate_interaction_csv(wall_id, max_posts=1)