import os
import re
from typing import List, Optional
from langchain_community.document_loaders import UnstructuredWordDocumentLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document


class UniversalDocumentProcessor:
    def __init__(
            self,
            file_path: str,
            chunk_size: int = 500,
            chunk_overlap: int = 100
    ):
        self.file_path = file_path
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        # Стоп-слова
        self.ignore_phrases = {
            "всі права захищені", "видавництво", "зміст", "передмова",
            "www.", "http", "isbn", "удк", "ббк", "©"
        }

    def _clean_text(self, text: str) -> str:
        if not text: return ""
        # Переноси слів
        text = re.sub(r'(\w+)-\n\s*(\w+)', r'\1\2', text)
        text = text.replace('\xa0', ' ')
        text = re.sub(r'\s+', ' ', text)
        return text.strip()

    def _is_garbage(self, text: str) -> bool:
        text_lower = text.lower()
        if len(text) < 5 or text.isdigit():
            return True
        for phrase in self.ignore_phrases:
            if phrase in text_lower:
                return True
        return False

    def _extract_citation_ref(self, text: str) -> Optional[str]:
        # Пошук розділу\параграфу
        patterns = [r'^§\s?\d+', r'^Розділ', r'^Глава', r'^Тема']
        for pat in patterns:
            if re.match(pat, text, re.IGNORECASE):
                return text
        return None

    def load_and_process(self) -> List[Document]:
        print(f"Початок обробки: {self.file_path}...")

        ext = os.path.splitext(self.file_path)[-1].lower()
        processed_docs = []
        raw_documents = []

        try:
            if ext == '.pdf':
                print("Формат PDF (PyPDFLoader)")
                loader = PyPDFLoader(self.file_path)
                raw_documents = loader.load()

            elif ext in ['.docx', '.doc']:
                print("Формат DOCX")
                loader = UnstructuredWordDocumentLoader(
                    self.file_path,
                    mode="elements",
                    strategy="fast"
                )
                raw_documents = loader.load()
            else:
                raise ValueError(f"Формат {ext} не підтримується.")

            current_section = "Вступ"
            citation_ref = "Загальний контекст"

            for doc in raw_documents:
                content = self._clean_text(doc.page_content)

                page_num = doc.metadata.get('page', doc.metadata.get('page_number'))
                page_info = f", Стор. {page_num + 1}" if page_num is not None else ""

                if self._is_garbage(content):
                    continue

                ref_candidate = self._extract_citation_ref(content)
                if ref_candidate:
                    current_section = ref_candidate
                    citation_ref = ref_candidate

                injected_content = f"Розділ: {current_section}\nТекст: {content}"

                new_doc = Document(
                    page_content=injected_content,
                    metadata={
                        "source": self.file_path,
                        "section": current_section,
                        "citation_ref": f"{citation_ref}{page_info}",
                        "page_number": str(page_num + 1) if page_num is not None else None
                    }
                )
                processed_docs.append(new_doc)

            # Чанкінг
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
                separators=["\n\n", "\n", ". ", " ", ""]
            )

            final_chunks = splitter.split_documents(processed_docs)
            print(f"Успіх! Оброблено {len(final_chunks)} чанків.")
            return final_chunks

        except Exception as e:
            print(f"Помилка обробки: {e}")
            return []


if __name__ == "__main__":
    pass
