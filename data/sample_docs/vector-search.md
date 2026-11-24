# Vector Search

Vector search finds the documents whose embeddings are closest to the embedding of a query. Instead of matching keywords, it matches meaning, so a query about "chunk size" can retrieve a passage that says "splitting documents into pieces" without sharing the exact words.

Exact nearest-neighbor search compares the query vector against every vector in the index. It is perfectly accurate but slow, because the work grows linearly with the number of documents, so it is only practical for small collections.

Approximate nearest neighbor search, or ANN, trades a small amount of accuracy for much faster queries. It organizes vectors into a structure that lets the search skip most of the index, which is what makes vector search usable at scale.

HNSW, short for Hierarchical Navigable Small World, is a popular ANN algorithm. It builds a layered graph where each vector is a node connected to its nearest neighbors, and a query walks the graph from a coarse top layer down to the detailed bottom layer.

An index stores vectors together with the metadata needed to use them, such as the source document and the chunk text. Building the index is a one-time cost, while querying it is the hot path, so index design optimizes for fast reads.

Top-k retrieval returns the k closest vectors to the query, where k is a setting the developer chooses. A small k returns only the most relevant chunks, while a large k returns more context at the cost of including less relevant material.

Chroma is a lightweight open-source vector database that runs locally with no server to manage. It stores embeddings in a persistent directory on disk, which makes it a good fit for a personal project or a prototype.

Metadata filtering narrows a vector search to a subset of the index before ranking by similarity. For example, a query can be restricted to chunks from a single source document or to documents published after a certain date.

Vector search struggles with exact terms such as product codes, names, and numbers, because embeddings blur distinct tokens that look similar. That weakness is why many systems combine vector search with keyword search in a hybrid retriever.

Relevance in vector search is only as good as the embeddings and the chunking behind it. If chunks are too large or the embedding model is a poor fit for the domain, even a perfect index will return the wrong passages.
