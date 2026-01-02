import os
import re
from typing import List, Dict, Set, Optional, Tuple
from langchain_core.documents import Document


class CitationManager:

    @staticmethod
    def _clean_filename(path: str) -> str:
        return os.path.basename(path)

    @staticmethod
    def _parse_citation_string(citation_string: str) -> Tuple[str, Optional[str]]:
        # Шукаємо: (Будь-що), (коми/пробіли), (Слово Стор/Page/стор крапка/без), (пробіли), (Будь-що до кінця)
        match = re.search(
            r'(.*),\s*(?:Стор\.?|Page|стор\.?)\s*(.*)',
            citation_string,
            re.IGNORECASE
        )

        if match:
            clean_title = match.group(1).strip()
            page_str = match.group(2).strip()
            return clean_title, page_str
        else:
            return citation_string, None

    @staticmethod
    def process_sources(docs: List[Document]) -> str:
        if not docs:
            return ""

        # Словник для агрегації: Ключ: (Назва розділу, Ім'я файлу) -> Значення: Множина сторінок
        grouped_sources = {}

        for doc in docs:
            raw_ref = doc.metadata.get('citation_ref', 'Загальний контекст')
            filepath = doc.metadata.get('source', 'Невідомий файл')
            filename = CitationManager._clean_filename(filepath)

            title, page = CitationManager._parse_citation_string(raw_ref)

            key = (title, filename)

            if key not in grouped_sources:
                grouped_sources[key] = set()

            if page:
                grouped_sources[key].add(page)

        output_lines = []

        sorted_keys = sorted(grouped_sources.keys(), key=lambda x: (x[1], x[0]))

        for title, filename in sorted_keys:
            pages = grouped_sources[(title, filename)]

            if pages:
                sorted_pages = sorted(
                    list(pages),
                    key=lambda x: int(x) if x.isdigit() else str(x)
                )
                pages_str = ", ".join(sorted_pages)

                line = f"{title} (Стор. {pages_str}) — [{filename}]"
            else:
                line = f"{title} — [{filename}]"

            output_lines.append(line)

        if not output_lines:
            return ""

        return "\n\n" + "=" * 30 + "\nДжерела:\n" + "\n".join(
            [f"{i}. {line}" for i, line in enumerate(output_lines, 1)]
        )

    @staticmethod
    def merge(answer: str, docs: List[Document]) -> str:
        citations = CitationManager.process_sources(docs)
        return f"{answer}{citations}"