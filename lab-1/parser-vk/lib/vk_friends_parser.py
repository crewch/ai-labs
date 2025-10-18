"""
vk_friends_parser.py

Модульный объектно-ориентированный парсер друзей ВКонтакте (VK).

Функции:
- Получение id по ссылке/короткому имени
- Выгрузка списка друзей пользователя (с глубиной: 1 - друзья, 2 - друзья друзей)
- Сбор метаданных пользователя, включая изображение профиля в io.BytesIO
- Построение графа (networkx) с атрибутами
- Сохранение/загрузка сериализованных данных (pickle, gzip)
- Вычисление центральностей: посредничество (betweenness), близость (closeness), собственный вектор (eigenvector)
- Простая модель прогноза дружбы (признаки: common neighbors, jaccard, adamic-adar, preferential attachment)
- CLI-обёртка для использования из консоли

Требования:
- Python 3.8+
- requests
- networkx
- scikit-learn (опционально для тренировки модели)

Пример запуска из консоли:
python vk_friends_parser.py fetch --id vk.com/durov --depth 2 --token YOUR_TOKEN --out data.pkl.gz --compute-centrality

Автор: скрипт-реализация для лабораторной работы.
"""

from __future__ import annotations

import argparse
import base64
import gzip
import io
import json
import pickle
import re
import time
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple, Iterable

import requests
import networkx as nx

import os

try:
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import roc_auc_score
    SKLEARN_AVAILABLE = True
except Exception:
    SKLEARN_AVAILABLE = False


VK_API_V = "5.199"
# DEFAULT_FIELDS = "sex,bdate,city,country,domain,photo_200_orig,photo_max_orig,relation"
DEFAULT_FIELDS = (
    "sex,bdate,city,country,home_town,domain,photo_200_orig,photo_max_orig,"
    "relation,relation_partner,relatives,schools,universities,education,"
    "career,occupation,site,timezone,contacts,connections,"
    "status,last_seen,followers_count,verified,"
    "interests,music,movies,tv,books,games,about,activities,"
    "quotes,personal,can_write_private_message,"
    "can_see_audio,can_send_friend_request"
)


@dataclass
class Person:
    id: int
    first_name: Optional[str] = None
    last_name: Optional[str] = None
    domain: Optional[str] = None
    photo_bytes: Optional[bytes] = None  # raw bytes (can be stored as-is in pickle)
    meta: Dict = None

    def to_dict(self) -> Dict:
        d = asdict(self)
        # Do not attempt to convert photo_bytes; it's binary and pickle will handle it.
        return d


class VKClient:
    """Небольшая обёртка над VK API.

    Требует access_token с правом friends (public API может возвращать неполные данные).
    Токен можно передать в конструктор или указать как переменную окружения VK_TOKEN.
    """

    def __init__(self, token: str, api_version: str = VK_API_V, session: Optional[requests.Session] = None):
        self.token = token
        self.v = api_version
        self.session = session or requests.Session()
        self.base = "https://api.vk.com/method/"

    def _call(self, method: str, params: Dict) -> Dict:
        url = self.base + method
        p = {**params, "access_token": self.token, "v": self.v}
        resp = self.session.get(url, params=p)
        resp.raise_for_status()
        data = resp.json()
        if "error" in data:
            raise RuntimeError(f"VK API error: {data['error']}")
        return data["response"]

    def resolve_screen_name(self, screen_name: str) -> Optional[int]:
        # screen_name can be 'durov' or full vk.com/durov
        m = re.match(r"^(https?://)?(www\.)?vk\.com/([A-Za-z0-9_.]+)$", screen_name)
        if m:
            screen_name = m.group(3)
        res = self._call("utils.resolveScreenName", {"screen_name": screen_name})
        if res and res.get("type") == "user":
            return res.get("object_id")
        return None

    def get_user(self, user_id: int) -> Dict:
        params = {"user_ids": user_id, "fields": DEFAULT_FIELDS}
        resp = self._call("users.get", params)
        if isinstance(resp, list):
            return resp[0]
        return resp

    def get_friends(self, user_id: int, fields: str = DEFAULT_FIELDS) -> List[int]:
        # returns list of friend ids (optionally extended info via users.get)
        params = {"user_id": user_id, "order": "hints", "fields": fields}
        resp = self._call("friends.get", params)
        # resp contains 'items' and 'count'
        return resp.get("items", [])

    def get_users_bulk(self, user_ids: List[int], fields: str = DEFAULT_FIELDS) -> List[Dict]:
        """Получить информацию о пользователях пакетами по 1000 id за раз."""
        results = []
        for i in range(0, len(user_ids), 500):
            chunk = user_ids[i:i + 500]
            params = {"user_ids": ",".join(map(str, chunk)), "fields": fields}
            try:
                resp = self._call("users.get", params)
                if isinstance(resp, list):
                    results.extend(resp)
            except Exception as e:
                print(f"Ошибка пакетного запроса {chunk[:3]}...: {e}")
            time.sleep(0.3)
        return results

    def download_photo_bytes(self, url: str) -> Optional[bytes]:
        if not url:
            return None
        try:
            r = self.session.get(url, timeout=10)
            r.raise_for_status()
            return r.content
        except Exception:
            return None


class VKFriendsParser:
    """Высокоуровневый интерфейс для сбора данных о друзьях и построения графа."""

    def __init__(self, vk_client: VKClient, save_photos: bool = True, rate_sleep: float = 0.37):
        self.vk = vk_client
        self.save_photos = save_photos
        self.rate_sleep = rate_sleep  # be polite
        self.people: Dict[int, Person] = {}
        self.edges: List[Tuple[int, int, Dict]] = []  # undirected edges with metadata

    def _add_person_from_vk(self, user: Dict) -> Person:
        uid = int(user["id"])
        if uid in self.people:
            return self.people[uid]

        photo_url = user.get("photo_max_orig") or user.get("photo_200_orig")
        photo_bytes = None
        photo_filename = None

        if self.save_photos and photo_url:
            photo_bytes = self.vk.download_photo_bytes(photo_url)
            if photo_bytes:
                os.makedirs("profile_photos", exist_ok=True)
                photo_filename = f"profile_photos/{uid}.jpg"
                with open(photo_filename, "wb") as f:
                    f.write(photo_bytes)
            time.sleep(self.rate_sleep)

        p = Person(
            id=uid,
            first_name=user.get("first_name"),
            last_name=user.get("last_name"),
            domain=user.get("domain"),
            photo_bytes=photo_bytes,  # для pickle остаётся
            meta={k: user.get(k) for k in user.keys()
                  if k not in ("first_name", "last_name", "id", "domain", "photo_max_orig", "photo_200_orig")}
        )

        # Дополнительно сохраняем имя файла и ссылку для JSON
        if photo_filename:
            p.meta["photo_file"] = photo_filename

        if photo_url:
            p.meta["photo_url"] = photo_url

        self.people[uid] = p
        return p

    def fetch_network(self, root_ids: Iterable[int], depth: int = 1, include_root_meta: bool = True) -> None:
        """Собирает друзей до глубины depth (1 - только друзья, 2 - друзья и друзья друзей).

        root_ids: iterable user ids to start
        depth: 1..2 (поддерживаем до 2, можно расширить)
        """
        if depth < 1:
            return
        frontier = set(int(x) for x in root_ids)
        visited = set()

        for d in range(depth):
            next_frontier = set()
            for uid in list(frontier):
                if uid in visited:
                    continue
                try:
                    # get friend IDs with partial info
                    items = self.vk.get_friends(uid)
                except Exception as e:
                    print(f"Ошибка при получении друзей {uid}: {e}")
                    items = []
                # items may be list of ids or list of dicts depending on API call
                friend_ids = []
                friend_dicts = []
                if items and isinstance(items[0], dict):
                    friend_dicts = items
                    friend_ids = [int(x["id"]) for x in items]
                else:
                    friend_ids = [int(x) for x in items]

                # optionally fetch metadata for the root node
                if include_root_meta and uid not in self.people:
                    try:
                        uinfo = self.vk.get_user(uid)
                        self._add_person_from_vk(uinfo)
                        time.sleep(self.rate_sleep)
                    except Exception:
                        pass

                # for each friend, fetch full metadata
                for fid in friend_ids:
                    try:
                        uinfo = self.vk.get_user(fid)
                        p = self._add_person_from_vk(uinfo)
                        # edge
                        self.edges.append((uid, fid, {}))
                    except Exception as e:
                        # If cannot fetch full info, still register id
                        if fid not in self.people:
                            self.people[fid] = Person(id=fid, meta={})
                        self.edges.append((uid, fid, {}))
                    time.sleep(self.rate_sleep)

                visited.add(uid)
                next_frontier.update(friend_ids)
            frontier = next_frontier - visited

    def fetch_network_fast(self, root_ids: Iterable[int], depth: int = 1, include_root_meta: bool = True) -> None:
        """Пакетный сбор друзей и друзей друзей."""
        if depth < 1:
            return

        frontier = set(int(x) for x in root_ids)
        visited = set()

        for d in range(depth):
            print(f"=== Глубина {d + 1}/{depth}, обрабатываем {len(frontier)} пользователей ===")
            next_frontier = set()

            for uid in list(frontier):
                if uid in visited:
                    continue

                try:
                    items = self.vk.get_friends(uid)
                except Exception as e:
                    print(f"Ошибка при получении друзей {uid}: {e}")
                    items = []

                friend_ids = []
                if items and isinstance(items[0], dict):
                    friend_ids = [int(x["id"]) for x in items]
                else:
                    friend_ids = [int(x) for x in items]

                # --- метаданные самого пользователя ---
                if include_root_meta and uid not in self.people:
                    try:
                        uinfo = self.vk.get_user(uid)
                        self._add_person_from_vk(uinfo)
                        time.sleep(self.rate_sleep)
                    except Exception:
                        pass

                # --- пакетное получение метаданных друзей ---
                if friend_ids:
                    friend_infos = self.vk.get_users_bulk(friend_ids)
                    found_ids = set()

                    for info in friend_infos:
                        p = self._add_person_from_vk(info)
                        self.edges.append((uid, p.id, {}))
                        found_ids.add(p.id)

                    # Добавляем оставшихся (если не удалось получить инфо)
                    for fid in set(friend_ids) - found_ids:
                        if fid not in self.people:
                            self.people[fid] = Person(id=fid, meta={})
                        self.edges.append((uid, fid, {}))

                    # добавляем во frontier
                    next_frontier.update(friend_ids)

                visited.add(uid)

            frontier = next_frontier - visited

    def build_graph(self, undirected: bool = True) -> nx.Graph:
        G = nx.Graph() if undirected else nx.DiGraph()
        for pid, person in self.people.items():
            attr = {**(person.meta or {}), "first_name": person.first_name, "last_name": person.last_name, "domain": person.domain}
            # don't put big binary in network attributes if undesired; but we will include size and a flag
            if person.photo_bytes:
                attr["photo_bytes_len"] = len(person.photo_bytes)
                attr["has_photo"] = True
            else:
                attr["has_photo"] = False
            G.add_node(pid, **attr)
        for a, b, meta in self.edges:
            if not G.has_node(a):
                G.add_node(a)
            if not G.has_node(b):
                G.add_node(b)
            G.add_edge(a, b, **(meta or {}))
        return G

    def save(self, path: str) -> None:
        payload = {"people": self.people, "edges": self.edges}
        with gzip.open(path, "wb") as f:
            pickle.dump(payload, f)

    def save_json(self, path: str) -> None:
        payload = {
            "people": [
                {
                    **{k: v for k, v in p.to_dict().items() if k != "photo_bytes"},
                    "photo_file": p.meta.get("photo_file"),
                    "photo_url": p.meta.get("photo_url")
                }
                for p in self.people.values()
            ],
            "edges": [list(e) for e in self.edges]
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, path: str, vk_client: Optional[VKClient] = None) -> "VKFriendsParser":
        with gzip.open(path, "rb") as f:
            payload = pickle.load(f)
        parser = cls(vk_client=vk_client or VKClient(token=""), save_photos=False)
        parser.people = payload.get("people", {})
        parser.edges = payload.get("edges", [])
        return parser

    # --- centrality calculations ---
    @staticmethod
    def compute_centralities(G: nx.Graph, subject_ids: Optional[Iterable[int]] = None) -> Dict[int, Dict[str, float]]:
        # If subject_ids provided, compute only for them in returned dict, but centralities need global computation
        # Compute betweenness, closeness, eigenvector (per connected component for eigenvector)
        bet = nx.betweenness_centrality(G, k=500, seed=42) if G.number_of_nodes() > 500 else nx.betweenness_centrality(G)
        clo = nx.closeness_centrality(G)
        eig = {}
        # eigenvector centrality can fail on disconnected graphs; compute per component
        for comp in nx.connected_components(G):
            sub = G.subgraph(comp)
            try:
                ev = nx.eigenvector_centrality_numpy(sub, max_iter=100, tol=1e-06)
            except Exception:
                try:
                    ev = nx.eigenvector_centrality(sub, max_iter=200)
                except Exception:
                    ev = {n: 0.0 for n in sub.nodes()}
            eig.update(ev)
        result = {}
        subjects = set(subject_ids) if subject_ids is not None else set(G.nodes())
        for n in subjects:
            result[n] = {"betweenness": float(bet.get(n, 0.0)), "closeness": float(clo.get(n, 0.0)), "eigenvector": float(eig.get(n, 0.0))}
        return result

    # --- link-prediction features ---
    @staticmethod
    def pair_features(G: nx.Graph, u: int, v: int) -> Dict[str, float]:
        # features: common_neighbors, jaccard, adamic_adar, preferential_attachment, degree_diff
        cn = len(list(nx.common_neighbors(G, u, v))) if G.has_node(u) and G.has_node(v) else 0
        jacc = 0.0
        try:
            jacc_gen = nx.jaccard_coefficient(G, [(u, v)])
            jacc = next(jacc_gen)[2]
        except Exception:
            jacc = 0.0
        aa = 0.0
        try:
            aa_gen = nx.adamic_adar_index(G, [(u, v)])
            aa = next(aa_gen)[2]
        except Exception:
            aa = 0.0
        pa = 0
        if G.has_node(u) and G.has_node(v):
            pa = G.degree(u) * G.degree(v)
        deg_diff = abs((G.degree(u) if G.has_node(u) else 0) - (G.degree(v) if G.has_node(v) else 0))
        return {"common_neighbors": cn, "jaccard": float(jacc), "adamic_adar": float(aa), "pref_attach": float(pa), "deg_diff": float(deg_diff)}

    def candidate_pairs(self, G: nx.Graph, nodes: Optional[Iterable[int]] = None) -> Iterable[Tuple[int, int]]:
        # Heuristic: consider pairs at distance 2 (friends of friends) as candidate positive links
        nodes = nodes or G.nodes()
        seen = set()
        for u in nodes:
            for v in G.nodes():
                if u == v:
                    continue
                if G.has_edge(u, v):
                    continue
                # consider only if they share at least one common neighbor
                if nx.node_connected_component(G, u) is not nx.node_connected_component(G, v):
                    # skip across components
                    continue
                if len(list(nx.common_neighbors(G, u, v))) > 0:
                    pair = (min(u, v), max(u, v))
                    if pair not in seen:
                        seen.add(pair)
                        yield pair

    def build_feature_matrix(self, G: nx.Graph, pairs: Iterable[Tuple[int, int]]) -> Tuple[List[List[float]], List[Tuple[int, int]]]:
        X = []
        idx = []
        for u, v in pairs:
            f = self.pair_features(G, u, v)
            X.append([f[k] for k in ("common_neighbors", "jaccard", "adamic_adar", "pref_attach", "deg_diff")])
            idx.append((u, v))
        return X, idx

    def train_link_predictor(self, G: nx.Graph, positive_edges: List[Tuple[int, int]], negative_edges: Optional[List[Tuple[int, int]]] = None) -> Optional[object]:
        """Train a simple logistic regression for link prediction. Returns trained model or None if sklearn missing.

        positive_edges: list of existing edges considered positive examples
        negative_edges: list of non-edges (if None, will sample random non-edges)
        """
        if not SKLEARN_AVAILABLE:
            print("scikit-learn not available: cannot train model")
            return None
        pos_pairs = [(min(a, b), max(a, b)) for a, b in positive_edges]
        if negative_edges is None:
            # sample non-edges
            non_edges = list(nx.non_edges(G))
            # sample size
            N = min(len(pos_pairs), max(100, len(pos_pairs)))
            neg_pairs = non_edges[:N]
        else:
            neg_pairs = [(min(a, b), max(a, b)) for a, b in negative_edges]
        pos_X, _ = self.build_feature_matrix(G, pos_pairs)
        neg_X, _ = self.build_feature_matrix(G, neg_pairs)
        X = pos_X + neg_X
        y = [1] * len(pos_X) + [0] * len(neg_X)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        clf = LogisticRegression(max_iter=1000)
        clf.fit(X_train, y_train)
        if len(X_test) > 0:
            y_prob = clf.predict_proba(X_test)[:, 1]
            try:
                auc = roc_auc_score(y_test, y_prob)
                print(f"Trained logistic regression, AUC={auc:.3f}")
            except Exception:
                pass
        return clf


# ---------------- CLI ----------------

def parse_args():
    p = argparse.ArgumentParser(description="VK friends parser and graph builder")
    sub = p.add_subparsers(dest="cmd")

    fetch = sub.add_parser("fetch", help="Fetch friends and build dataset")
    fetch.add_argument("--id", "-i", required=True, help="VK id or profile link (e.g. vk.com/durov or numeric id)")
    fetch.add_argument("--token", "-t", required=True, help="VK API access token")
    fetch.add_argument("--depth", "-d", type=int, default=1, choices=[1, 2], help="Depth of fetch: 1 or 2")
    fetch.add_argument("--out", "-o", default="vk_data.pkl.gz", help="Output gzipped pickle file")
    fetch.add_argument("--no-photos", action="store_true", help="Do not download profile photos")
    fetch.add_argument("--compute-centrality", action="store_true", help="Compute centralities and print for root user")

    info = sub.add_parser("info", help="Print summary of saved dataset")
    info.add_argument("--file", "-f", required=True, help="File to inspect")

    build = sub.add_parser("build-graph", help="Load dataset and build graph file")
    build.add_argument("--file", "-f", required=True, help="File with dataset")
    build.add_argument("--out-graph", "-g", default="graph.gpickle", help="Output graph in gpickle format")

    return p.parse_args()


def _to_int_id(vk_id_or_link: str, vk: VKClient) -> int:
    if re.match(r"^\d+$", vk_id_or_link):
        return int(vk_id_or_link)
    # maybe vk.com/...
    if re.search(r"vk\.com", vk_id_or_link):
        try:
            m = re.search(r"vk\.com/([A-Za-z0-9_.]+)$", vk_id_or_link)
            if m:
                name = m.group(1)
                rid = vk.resolve_screen_name(name)
                if rid:
                    return int(rid)
        except Exception:
            pass
    # fallback: try resolve as screen name
    rid = vk.resolve_screen_name(vk_id_or_link)
    if rid:
        return int(rid)
    raise ValueError(f"Cannot resolve id from {vk_id_or_link}")


def main():
    args = parse_args()
    if args.cmd == "fetch":
        vk = VKClient(token=args.token)
        root_id = _to_int_id(args.id, vk)
        parser = VKFriendsParser(vk_client=vk, save_photos=not args.no_photos)
        print(f"Fetching network for {root_id} depth={args.depth} ...")
        parser.fetch_network([root_id], depth=args.depth)
        parser.save(args.out)
        print(f"Saved to {args.out}. nodes={len(parser.people)} edges={len(parser.edges)}")
        if args.compute_centrality:
            G = parser.build_graph()
            cent = parser.compute_centralities(G, subject_ids=[root_id])
            print("Centralities for root:")
            print(cent.get(root_id))
    elif args.cmd == "info":
        parser = VKFriendsParser.load(args.file)
        print(f"Loaded {args.file}: nodes={len(parser.people)} edges={len(parser.edges)}")
    elif args.cmd == "build-graph":
        parser = VKFriendsParser.load(args.file)
        G = parser.build_graph()
        nx.write_gpickle(G, args.out_graph)
        print(f"Graph written to {args.out_graph}. nodes={G.number_of_nodes()} edges={G.number_of_edges()}")
    else:
        print("Run with --help to see commands")


if __name__ == "__main__":
    main()
