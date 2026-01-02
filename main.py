import os
import sys
from dotenv import load_dotenv

from ingestion import UniversalDocumentProcessor
from vector_store import VectorStoreManager
from rag_engine import ProfessionalRAGEngine
from citation_handler import CitationManager

load_dotenv()

FILE_NAME = "your_file_name"
DATA_DIR = "data"
FILE_PATH = os.path.join(DATA_DIR, FILE_NAME)


COLLECTION_NAME = "educational_material"


def main():
    print("Ініціалізація RAG системи...")

    if not os.getenv("OPENAI_API_KEY"):
        print("Помилка: Не знайдено OPENAI_API_KEY.")
        print(" Створіть файл .env і додайте туди: OPENAI_API_KEY=sk-...")
        return

    print(f"Підключення до бази знань '{COLLECTION_NAME}'...")

    vs_manager = VectorStoreManager(
        collection_name=COLLECTION_NAME,
        model_type="openai"
    )
    try:
        vs_manager.load_index()
        print("База знань успішно завантажена з диска.")
    except FileNotFoundError:
        print("База не знайдена. Починаємо створення нової...")

        if not os.path.exists(FILE_PATH):
            print(f"Файл підручника не знайдено за шляхом: {FILE_PATH}")
            print(f"Будь ласка, покладіть файл у папку '{DATA_DIR}' та перевірте змінну FILE_NAME.")
            return

        print(f"Обробка файлу: {FILE_NAME}")
        processor = UniversalDocumentProcessor(
            file_path=FILE_PATH,
            chunk_size=500,
            chunk_overlap=100
        )
        chunks = processor.load_and_process()

        if not chunks:
            print("Не вдалося отримати текст з файлу. Перевірте формат.")
            return

        vs_manager.create_index(chunks)

    print("Налаштування AI-асистента...")
    rag_engine = ProfessionalRAGEngine(
        vector_store=vs_manager,
        model_name="gpt-4o",
        k_retrieval=5
    )

    print("\n" + "=" * 50)
    print(f"RAG-Чат готовий! (Файл: {FILE_NAME})")
    print("Введіть 'exit', 'quit' або 'вихід' для завершення.")
    print("=" * 50 + "\n")

    while True:
        try:
            query = input("Ваш запит: ").strip()

            if not query:
                continue

            if query.lower() in ['exit', 'quit', 'вихід']:
                print("До побачення!")
                break

            result = rag_engine.get_answer(query)

            final_output = CitationManager.merge(
                answer=result['answer'],
                docs=result['source_documents']
            )

            print(f"\nБот:\n{final_output}")
            print("-" * 50 + "\n")

        except KeyboardInterrupt:
            print("\nРоботу перервано користувачем.")
            break
        except Exception as e:
            print(f"\nВиникла непередбачувана помилка: {e}")


if __name__ == "__main__":
    main()
