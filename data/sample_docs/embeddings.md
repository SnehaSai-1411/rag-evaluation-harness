# Embeddings

An embedding is a list of numbers, called a vector, that represents the meaning of a piece of text. Similar texts produce vectors that point in similar directions, which is what makes semantic search possible.

Embedding models are trained so that texts with similar meanings end up close together in vector space. The model learns this from large amounts of paired text, such as questions and their answers or sentences and their paraphrases.

The dimensionality of an embedding is the number of values in its vector. Common sizes range from 384 dimensions for small local models to 1536 dimensions for large API models, and larger vectors can capture finer distinctions at the cost of more storage and slower search.

Cosine similarity measures the angle between two vectors and is the standard way to compare embeddings. A cosine similarity of 1 means the vectors point in exactly the same direction, while 0 means they are unrelated.

Sentence-transformers is a Python library for running embedding models locally on CPU. A popular small model is all-MiniLM-L6-v2, which produces 384-dimensional embeddings and runs fast enough for a personal project without a GPU.

Local embedding models have no per-request cost and keep your documents on your own machine. The trade-off is lower quality than the largest API models and the need to manage the model download and the runtime yourself.

API embedding models are hosted by providers, and you pay per token. They are usually more accurate than small local models, but every document and query leaves your machine, and costs grow with index size.

The same embedding model must be used for indexing and for querying. If documents are embedded with one model and the question with another, the vectors live in different spaces and similarity scores become meaningless.

Embedding quality depends on the domain. A model trained on general web text may struggle with specialized vocabulary, so for a niche corpus it is worth testing a domain-tuned model or fine-tuning on in-domain pairs.

Embeddings should be normalized before storage when cosine similarity is the comparison metric. Normalizing each vector to unit length turns cosine similarity into a simple dot product, which is faster to compute at query time.
