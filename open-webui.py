import requests
import logging
import pandas as pd

logging.basicConfig(level=logging.INFO)

class OpenWebUIClient:
    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip('/')
        self.token = None
        self.headers: dict = {}

    def sign_in(self, email: str, password: str):
        """
        Authentifie l’utilisateur et récupère un token JWT.
        Endpoint: POST /api/v1/auths/signin :contentReference[oaicite:0]{index=0}
        """
        url = f"{self.base_url}/api/v1/auths/signin"
        payload = {"email": email, "password": password}
        resp = requests.post(url, json=payload)
        resp.raise_for_status()
        self.token = resp.json().get("token")
        if not self.token:
            raise ValueError("Token non trouvé dans la réponse")
        self.headers = {'Authorization': f'Bearer {self.token}'}
        logging.info("Authentification réussie")

    def create_user(self, name: str, email: str, password: str, role: str = 'admin'):
        """
        Crée un nouvel utilisateur (admin ou user).
        Endpoint (non officiell): POST /api/v1/users :contentReference[oaicite:1]{index=1}
        """
        url = f"{self.base_url}/api/v1/users"
        payload = {"name": name, "email": email, "password": password, "role": role}
        resp = requests.post(
            url,
            headers={**self.headers, 'Content-Type': 'application/json'},
            json=payload
        )
        resp.raise_for_status()
        logging.info(f"Utilisateur {email} créé")
        return resp.json()

    def pull_model(self, model_name: str):
        """
        Télécharge/pull un modèle via le proxy Ollama.
        Proxy Ollama: /ollama/api/pull :contentReference[oaicite:2]{index=2}
        """
        url = f"{self.base_url}/ollama/api/pull"
        payload = {"model": model_name}
        resp = requests.post(
            url,
            headers={**self.headers, 'Content-Type': 'application/json'},
            json=payload
        )
        resp.raise_for_status()
        logging.info(f"Modèle {model_name} en cours de téléchargement")
        return resp.json()

    def create_collection(self, name: str, description: str = ''):
        """
        Crée une nouvelle collection de connaissance.
        Endpoint: POST /api/v1/knowledge/create :contentReference[oaicite:3]{index=3}
        """
        url = f"{self.base_url}/api/v1/knowledge/create"
        payload = {
            "name": name,
            "description": description,
            "data": {},
            "access_control": {}
        }
        resp = requests.post(
            url,
            headers={**self.headers, 'Content-Type': 'application/json'},
            json=payload
        )
        resp.raise_for_status()
        collection = resp.json()
        logging.info(f"Collection '{name}' créée (ID: {collection['id']})")
        return collection

    def upload_file(self, file_path: str):
        """
        Upload d’un fichier pour RAG.
        Endpoint: POST /api/v1/files/ :contentReference[oaicite:4]{index=4}
        """
        url = f"{self.base_url}/api/v1/files/"
        with open(file_path, 'rb') as f:
            files = {'file': f}
            resp = requests.post(
                url,
                headers={'Authorization': self.headers['Authorization']},
                files=files
            )
        resp.raise_for_status()
        file_info = resp.json()
        logging.info(f"Fichier {file_path} uploadé (ID: {file_info['id']})")
        return file_info

    def upload_content(self, content: str, filename: str):
        """
        Upload de contenu texte direct (utile pour un CSV transformé).
        Même endpoint que upload_file :contentReference[oaicite:5]{index=5}
        """
        url = f"{self.base_url}/api/v1/files/"
        files = {'file': (filename, content)}
        resp = requests.post(
            url,
            headers={'Authorization': self.headers['Authorization']},
            files=files
        )
        resp.raise_for_status()
        file_info = resp.json()
        logging.info(f"Contenu uploadé en tant que {filename} (ID: {file_info['id']})")
        return file_info

    def add_file_to_collection(self, collection_id: str, file_id: str):
        """
        Ajoute un fichier à une collection existante.
        Endpoint: POST /api/v1/knowledge/{id}/file/add :contentReference[oaicite:6]{index=6}
        """
        url = f"{self.base_url}/api/v1/knowledge/{collection_id}/file/add"
        payload = {"file_id": file_id}
        resp = requests.post(
            url,
            headers={**self.headers, 'Content-Type': 'application/json'},
            json=payload
        )
        resp.raise_for_status()
        logging.info(f"Fichier {file_id} ajouté à la collection {collection_id}")
        return resp.json()

    def chat_with_collection(self, model: str, query: str, collection_id: str):
        """
        Lance une requête RAG en se basant sur une collection.
        Endpoint: POST /api/chat/completions :contentReference[oaicite:7]{index=7}
        """
        url = f"{self.base_url}/api/chat/completions"
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": query}],
            "files": [{"type": "collection", "id": collection_id}]
        }
        resp = requests.post(
            url,
            headers={**self.headers, 'Content-Type': 'application/json'},
            json=payload
        )
        resp.raise_for_status()
        return resp.json()

    def evaluate_rag(self, model: str, eval_set: list[dict], collection_id: str):
        """
        Parcourt le jeu d'évaluation et renvoie requête, réponse et attendu.
        """
        results = []
        for item in eval_set:
            query = item["query"]
            expected = item["expected_answer"]
            response = self.chat_with_collection(model, query, collection_id)
            content = response["choices"][0]["message"]["content"]
            results.append({
                "query": query,
                "response": content,
                "expected": expected
            })
        return results


if __name__ == "__main__":
    # Exemple d’utilisation
    client = OpenWebUIClient("http://localhost:3000")

    # 1. Authentification initiale
    client.sign_in("admin_init@example.com", "init_password")

    # 2. Création d’un nouvel admin
    new_admin = client.create_user(
        name="Mon Admin",
        email="admin2@example.com",
        password="securePass123",
        role="admin"
    )

    # 3. Pull du modèle Quwen2 0.5B
    client.pull_model("quwen2:0.5b")

    # 4. Création d’une collection pour la KB
    coll = client.create_collection(name="Ma_KB_CSV", description="Base de connaissance issue d’un CSV")

    # 5. Ingestion de la KB depuis un CSV
    df = pd.read_csv("kb.csv")  # colonnes : id, content
    mapping = {}
    for _, row in df.iterrows():
        file_info = client.upload_content(
            content=row["content"],
            filename=f"{row['id']}.txt"
        )
        client.add_file_to_collection(coll["id"], file_info["id"])
        mapping[row["id"]] = file_info["id"]

    # 6. Benchmark RAG sur un jeu d’évaluation
    eval_df = pd.read_csv("eval_set.csv")  # colonnes : query,id_doc_rel,expected_answer
    eval_list = eval_df.rename(columns={"id_doc_rel": "relevant_id"}).to_dict(orient="records")
    results = client.evaluate_rag("quwen2:0.5b", eval_list, coll["id"])

    # Affichage des résultats
    for r in results:
        print(f"Q: {r['query']}\nR: {r['response']}\nAttendu: {r['expected']}\n---")
