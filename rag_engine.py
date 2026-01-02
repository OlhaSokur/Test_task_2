import os
import logging
from typing import List, Dict, Any
import tiktoken

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from langchain_core.documents import Document
from openai import APIConnectionError, RateLimitError, APIError

from vector_store import VectorStoreManager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ProfessionalRAGEngine:
    def __init__(
            self,
            vector_store: VectorStoreManager,
            model_name: str = "gpt-4o",
            k_retrieval: int = 5,
            max_context_tokens: int = 4000
    ):

        self.vector_store = vector_store
        self.k = k_retrieval
        self.model_name = model_name
        self.max_context_tokens = max_context_tokens

        # Перевірка API ключа
        if not os.getenv("OPENAI_API_KEY"):
            raise ValueError("Помилка: OPENAI_API_KEY не знайдено.")

        # Ініціалізація LLM
        self.llm = ChatOpenAI(
            model=self.model_name,
            temperature=0,
            request_timeout=20
        )

        prompt_template = self._create_prompt_template()
        self.chain = prompt_template | self.llm | StrOutputParser()

        try:
            self.tokenizer = tiktoken.encoding_for_model(model_name)
        except KeyError:
            self.tokenizer = tiktoken.get_encoding("cl100k_base")

    def _create_prompt_template(self) -> ChatPromptTemplate:
        template = """
        РОЛЬ:
        Ти — інтелектуальний аналітик-експерт, спеціаліст із опрацювання наукової та технічної документації. Твоє завдання — надавати точні, структуровані та обґрунтовані відповіді на основі наданих фрагментів тексту.

        КОНТЕКСТ ДЛЯ АНАЛІЗУ:
        {context}

        ЗАВДАННЯ:
        Дай відповідь на запитання: "{question}"

        СУВОРІ ПРАВИЛА РОБОТИ:
        1. ОБМЕЖЕННЯ ЗНАНЬ: Використовуй виключно наданий КОНТЕКСТ. Якщо у тексті немає прямої відповіді, напиши: "На жаль, у наданих документах немає інформації для відповіді на це запитання". Не додавай жодних фактів "від себе" або із зовнішніх джерел.
        2. ЦИТУВАННЯ: Кожне твердження у твоїй відповіді повинно мати посилання на джерело. Використовуй формат [Джерело: Назва/Розділ, Стор. Х]. Став посилання в кінці речення або абзацу, до якого воно відноситься.
        3. СТРУКТУРА: 
           - Якщо питання передбачає опис процесу або перелік, використовуй марковані списки.
           - Якщо інформація суперечлива, вкажи це, наводячи різні точки зору з контексту.
        4. ТОН: Дотримуйся офіційно-ділового, академічного стилю. Уникай вступних фраз типу "Згідно з текстом..." або "На основі наданих даних...". Переходь одразу до суті.
        5. МОВНИЙ РЕЖИМ: Відповідай тією ж мовою, якою поставлене запитання (українською).

        ЯКЩО ВІДПОВІДЬ ЗНАЙДЕНО:
        Сформуй чітку відповідь, де кожен факт підкріплений посиланням.

        ЯКЩО ВІДПОВІДЬ НЕ ЗНАЙДЕНО:
        Напиши лише фразу про відсутність інформації та (опціонально) коротко вкажи, про що саме йдеться у доступному контексті.
        """
        return ChatPromptTemplate.from_template(template)

    def _format_docs(self, docs: List[Document]) -> str:
        formatted_parts = []
        for doc in docs:
            ref = doc.metadata.get('citation_ref', 'Unknown Source')
            content = doc.page_content.replace("\n", " ")
            formatted_parts.append(f"[Джерело: {ref}] -\n{content}")
        return "\n\n".join(formatted_parts)

    def _safe_context_builder(self, docs: List[Document]) -> str:
        formatted_context = self._format_docs(docs)
        token_count = len(self.tokenizer.encode(formatted_context))

        if token_count <= self.max_context_tokens:
            return formatted_context

        logger.warning(f"Ліміт контексту перевищено ({token_count} > {self.max_context_tokens}). Прибираємо чанки...")

        while docs and token_count > self.max_context_tokens:
            docs.pop()
            formatted_context = self._format_docs(docs)
            token_count = len(self.tokenizer.encode(formatted_context))

        return formatted_context

    def get_answer(self, query: str) -> Dict[str, Any]:
        try:
            docs = self.vector_store.search_similarity(query, k=self.k, threshold=0.3)
            # Конвертуємо (Document, score) -> [Document]
            docs = [doc for doc, _ in docs]
        except Exception as e:
            logger.error(f"Помилка пошуку: {e}")
            return {"answer": "Помилка доступу до бази знань.", "source_documents": []}

        if not docs:
            return {
                "answer": "Я не знайшов інформації за вашим запитом у завантажених документах.",
                "source_documents": []
            }

        safe_context = self._safe_context_builder(docs)

        try:
            print(f"Надсилаємо запит до LLM (Довжина контексту: {len(safe_context)} чанок)...")

            response_text = self.chain.invoke({
                "context": safe_context,
                "question": query
            })

            return {
                "answer": response_text,
                "source_documents": docs
            }

        except RateLimitError:
            return {"answer": "Перевищено ліміт запитів до OpenAI. Спробуйте пізніше.", "source_documents": []}
        except APIConnectionError:
            return {"answer": "Проблеми з підключенням до сервера AI. Перевірте інтернет.", "source_documents": []}
        except Exception as e:
            logger.error(f"LLM Generation Error: {e}")
            return {"answer": f"Сталася технічна помилка при генерації відповіді: {str(e)}", "source_documents": []}