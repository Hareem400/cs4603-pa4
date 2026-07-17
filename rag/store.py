"""Vector Search retriever factory (Task 1.4 support / rag/store.py)."""

from __future__ import annotations

from functools import lru_cache

from config import get_settings

CITATION_COLUMNS = ["chunk_id", "source", "page"]


@lru_cache(maxsize=1)
def get_vector_store():
    """Return a DatabricksVectorSearch handle over the configured index."""
    from databricks.sdk import WorkspaceClient
    from databricks_langchain import DatabricksVectorSearch

    s = get_settings()
    if not s["vs_endpoint"] or not s["vs_index"]:
        raise OSError(
            "VECTOR_SEARCH_ENDPOINT and VECTOR_SEARCH_INDEX must be set "
            "(Task 0.3) before the retriever can be built."
        )

    # Explicit workspace_client so credentials (host + PAT) are always passed
    # through, instead of relying on notebook auto-detection (which fails
    # outside a Databricks notebook, e.g. in CI or the serving container).
    workspace_client = WorkspaceClient(
        host=s["host"], token=s["token"], auth_type="pat"
    )

    return DatabricksVectorSearch(
        endpoint=s["vs_endpoint"],
        index_name=s["vs_index"],
        columns=CITATION_COLUMNS,
        workspace_client=workspace_client,
    )


def get_retriever(k: int = 4):
    """Return a top-k retriever over the Vector Search index."""
    return get_vector_store().as_retriever(search_kwargs={"k": k})