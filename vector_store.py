import os
import shutil
from typing import List, Tuple, Optional
from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings


class VectorStoreManager:
    def __init__(
            self,
            collection_name: str = "knowledge_base",
            model_type: str = "openai",
            base_persist_dir: str = "./chroma_storage",
            api_key: Optional[str] = None
    ):
        self.collection_name = collection_name
        self.model_type = model_type.lower()

        # Налаштування моделі ембеддінгів
        if self.model_type == "openai":
            key = api_key or os.getenv("OPENAI_API_KEY")
            if not key:
                raise ValueError("Помилка: Не вказано OPENAI_API_KEY для моделі OpenAI.")

            self.embedding_model = OpenAIEmbeddings(
                model="text-embedding-3-small",
                api_key=key
            )
            model_dir = "openai_v3"

        elif self.model_type == "hf":
            self.embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
            model_dir = "hf_minilm"

        else:
            raise ValueError(f"Невідомий тип моделі '{model_type}'. Використовуйте 'openai' або 'hf'.")

        self.persist_directory = os.path.join(base_persist_dir, collection_name, model_dir)
        self.vector_db = None

    def create_index(self, documents: List[Document], force_reset: bool = False) -> None:
        if not documents:
            print("Увага: Отримано порожній список документів. Індексацію пропущено.")
            return

        if force_reset and os.path.exists(self.persist_directory):
            try:
                shutil.rmtree(self.persist_directory)
                print(f"Директорію очищено: {self.persist_directory}")
            except OSError as e:
                print(f"Не вдалося очистити директорію: {e}")

        print(f"Початок індексації {len(documents)} чанків у колекцію '{self.collection_name}'...")

        # Створення бази з явною метрикою 'cosine'
        self.vector_db = Chroma.from_documents(
            documents=documents,
            embedding=self.embedding_model,
            persist_directory=self.persist_directory,
            collection_name=self.collection_name,
            collection_metadata={"hnsw:space": "cosine"}
        )
        print(f"Індексацію завершено. База збережена в: {self.persist_directory}")

    def load_index(self) -> None:
        if not os.path.exists(self.persist_directory):
            raise FileNotFoundError(f"База даних не знайдена за шляхом: {self.persist_directory}")

        self.vector_db = Chroma(
            persist_directory=self.persist_directory,
            embedding_function=self.embedding_model,
            collection_name=self.collection_name,
            collection_metadata={"hnsw:space": "cosine"}
        )
        print(f"Базу даних '{self.collection_name}' успішно завантажено.")

    def search_similarity(self, query: str, k: int = 5, threshold: float = 0.0) -> List[Tuple[Document, float]]:
        if not self.vector_db:
            self.load_index()

        results_with_score = self.vector_db.similarity_search_with_score(query, k=k)

        filtered_results = []
        for doc, distance in results_with_score:
            similarity = 1.0 - distance

            if similarity >= threshold:
                filtered_results.append((doc, similarity))

        return filtered_results

    def get_retriever(self, k: int = 5):
        if not self.vector_db:
            self.load_index()

        return self.vector_db.as_retriever(
            search_type="similarity",
            search_kwargs={"k": k}
        )