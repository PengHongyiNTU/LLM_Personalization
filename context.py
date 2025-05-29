from abc import ABC, abstractmethod
from typing import Any, Optional, List, Set, Dict, Literal
from functools import lru_cache
import json
import os
import pickle
from collections import defaultdict
from langchain_text_splitters import RecursiveJsonSplitter
from langchain_community.vectorstores import FAISS
import getpass


class BaseContextLoader(ABC):
    @abstractmethod
    def get(self, query: str, **kwargs) -> Any:
        pass

    @abstractmethod
    def close(self) -> None:
        pass

    def __enter__(self) -> "BaseContextLoader":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()


class RawPersonaMemContextLoader(BaseContextLoader):
    def __init__(
        self,
        jsonl_path: str,
        index_path: Optional[str] = None,
        cache_size: int = 128,
    ):
        self.jsonl_path = jsonl_path
        self.index_path = index_path
        self._file_ctx = open(self.jsonl_path, "r", encoding="utf-8")

        # Load or build index
        if self.index_path and os.path.exists(self.index_path):
            with open(self.index_path, "rb") as f:
                self._index = pickle.load(f)
        else:
            self._index = self._build_index()
            if self.index_path:
                with open(self.index_path, "wb") as f:
                    pickle.dump(self._index, f)

        # Wrap the raw loader in an LRU cache
        self._load_cached = lru_cache(maxsize=cache_size)(self._load)

    def _build_index(self) -> dict[str, int]:
        idx = {}
        with open(self.jsonl_path, "r", encoding="utf-8") as f:
            while True:
                offset = f.tell()
                line = f.readline()
                if not line:
                    break
                key = next(iter(json.loads(line).keys()))
                idx[key] = offset
        return idx

    def _load(self, key: str) -> Any:
        if key not in self._index:
            raise KeyError(f"Context ID {key!r} not found")
        self._file_ctx.seek(self._index[key])
        line = self._file_ctx.readline()
        return json.loads(line)[key]

    def get(self, query: str, **kwargs) -> Any:
        end_index: Optional[int] = kwargs.pop("end_index", None)
        context = self._load_cached(query)
        if end_index is not None:
            if end_index > len(context):
                msg = (
                    f"end_index {end_index} exceeds",
                    f"the context length {len(context)}",
                )
                raise ValueError(msg)
            context = context[:end_index]
        return context

    def close(self) -> None:
        self._file_ctx.close()


class GroupedPersonaMemContextLoader(RawPersonaMemContextLoader):
    def __init__(
        self,
        jsonl_path: str,
        index_path: Optional[str] = None,
        cache_size: int = 128,
        embedding_provider: Literal["ollama", "openai"] = "ollama",
        embedding_model: Optional[str] = None,
        max_chunk_size: int = 512,
        top_k: int = 5,
        attach_profile: bool = True,
    ):
        super().__init__(jsonl_path, index_path, cache_size)
        self._session_to_persona: Dict[str, str] = {}
        self._persona_to_session: Dict[str, List] = defaultdict(list)
        self._group_by_persona()

        if embedding_provider == "ollama":
            from langchain_ollama import OllamaEmbeddings

            if not embedding_model:
                embedding_model = "mxbai-embed-large"
            self.embedding_model = OllamaEmbeddings(model=embedding_model)
        elif embedding_provider == "openai":
            if not os.getenv("OPENAI_API_KEY"):
                os.environ["OPENAI_API_KEY"] = getpass.getpass(
                    prompt="Please enter your OpenAI API key: "
                )
            from langchain_openai.embeddings import OpenAIEmbeddings

            if not embedding_model:
                embedding_model = "text-embedding-ada-002"
            self.embedding_model = OpenAIEmbeddings(model=embedding_model)
        else:
            raise ValueError(
                f"Unsupported embedding provider: {embedding_provider}"
            )

        self.splitter = RecursiveJsonSplitter(
            max_chunk_size=max_chunk_size,
        )
        # use in memory vector db for now
        self.vector_dbs = {}.fromkeys(self._persona_to_session.keys(), None)
        self.top_k = top_k
        self.attach_profile = attach_profile

    def _get_persona_key(self, message: str | dict) -> str:
        if isinstance(message, dict):
            persona_key = message.get("content", "")
        if isinstance(message, str):
            persona_key = message
        norm = " ".join(persona_key.strip().split())
        return norm

    def _group_by_persona(self):
        for session_id, offset in self._index.items():
            self._file_ctx.seek(offset)
            messages = json.loads(self._file_ctx.readline())[session_id]
            if not messages:
                continue
            persona_key = self._get_persona_key(messages[0])
            self._session_to_persona[session_id] = persona_key
            self._persona_to_session[persona_key].append(session_id)

    def _get_grouped_messages(self, session_id: str) -> List[dict]:
        persona = self._session_to_persona.get(session_id, None)
        if persona is None:
            raise KeyError(f"No matced persona for session_id {session_id}")
        seen: Set[str] = set()
        histories: List[dict] = []
        session_ids = self._persona_to_session[persona]

        for session_id in session_ids:
            messages = self._load(session_id)
            for msg in messages:
                fingerprint = json.dumps(msg, separators=(",", ":"))
                if fingerprint not in seen:
                    seen.add(fingerprint)
                    histories.append(msg)
        return histories

    def _setup_db(self, session_id: str, histories: List[dict]):
        if session_id in self.vector_dbs:
            return self.vector_dbs[session_id]
        else:
            docs = self.splitter.split_text(histories, convert_lists=True)
            db = FAISS.from_texts(
                docs,
                self.embedding_model,
                metadatas=[
                    {
                        "session_id": session_id,
                    }
                    for _ in docs
                ],
            )
            self.vector_dbs[session_id] = db
            return db

    def get_db(self, session_id: str) -> FAISS:
        if session_id not in self.vector_dbs:
            histories = self._get_grouped_messages(session_id)
            db = self._setup_db(session_id, histories)
            return db
        else:
            return self.vector_dbs[session_id]

    def get(self, query: str, **kwargs) -> Any:
        session_id: Optional[str] = kwargs.pop("session_id", None)
        top_k: Optional[int] = kwargs.pop("top_k", self.top_k)
        if session_id is None:
            raise ValueError("session_id is required")
        if session_id not in self._session_to_persona:
            raise KeyError(f"session_id {session_id} not found")
        db = self.get_db(session_id)
        retrieved = db.similarity_search(query, k=top_k)
        # concatenate the as whole list
        # retroved content is a str
        # need to serialize it
        context = []
        for doc in retrieved:
            content = doc.page_content
            doc_dict = json.loads(content)
            for _, value in doc_dict.items():
                if isinstance(value, list):
                    context.extend(value)
                else:
                    context.append(value)
        if self.attach_profile:
            persona = self._session_to_persona[session_id]
            context = [{"role": "system", "content": persona}] + context
        return context
