# Retrieval and Evaluation

Retrieval is the step in a RAG system that selects which chunks to show the language model. Its quality sets an upper bound on answer quality, because the model can only use the context it is given.

Top-k is the number of chunks a retriever returns for each query. Increasing k raises recall, since more of the relevant material is included, but it lowers precision, since more irrelevant material is included too.

Reranking runs a second, more expensive model over the retrieved chunks to reorder them by relevance. A typical pipeline retrieves twenty candidates with fast vector search and then reranks them down to the best five for the language model.

A golden dataset is a small set of questions with known correct answers and the source passages that support them. It is the ground truth an evaluation harness measures against, and it should be versioned in the repository like code.

Context precision measures the fraction of retrieved chunks that are actually relevant to the question. High precision means the model sees little noise, which reduces the chance of a confused or hallucinated answer.

Context recall measures the fraction of the relevant material that the retriever found. High recall means the answer is present in the retrieved context, even if it is buried among less relevant chunks.

Faithfulness checks whether the answer is supported by the retrieved context. An answer is faithful when every claim in it can be traced back to a retrieved passage, and unfaithful when it adds facts the context never stated.

Answer relevance measures how well the answer addresses the question that was asked. An answer can be faithful to the context and still score low on relevance if it answers a different question than the one the user posed.

An LLM-as-judge evaluation asks a language model to score an answer against a rubric. It catches nuance that token-overlap metrics miss, but it costs money per evaluation and its scores can vary between runs.

Evaluation belongs in continuous integration, so that a change to chunking, retrieval, or prompting cannot silently make answers worse. A CI gate that fails the build when mean scores drop below a threshold turns evaluation from a one-off experiment into a regression test.
