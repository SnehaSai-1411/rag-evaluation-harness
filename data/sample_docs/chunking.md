# Chunking

Chunking splits long documents into smaller pieces so a retrieval system can return the most relevant passage instead of an entire file. Each chunk becomes one unit that is embedded and stored in the vector index, and the chunk size controls the trade-off between precision and context.

Fixed-size chunking cuts text every N characters regardless of sentence or paragraph boundaries. It is the simplest strategy to implement, but it can split a sentence in half and leave a chunk that starts mid-thought.

Recursive chunking tries a list of separators in order, such as double newlines, single newlines, spaces, and finally individual characters. It keeps paragraphs and sentences intact whenever possible and only falls back to smaller splits when a piece is still too long.

Overlap repeats the last N characters of one chunk at the start of the next chunk. Overlap preserves context across chunk boundaries, which helps when an answer spans the split point, but it increases the total number of chunks and the size of the index.

Small chunks, around 200 to 400 characters, give precise retrieval because each chunk covers a narrow topic. The risk is that a chunk may be too short to contain a complete answer, so the language model receives a fragment instead of a full explanation.

Large chunks, around 800 to 1500 characters, carry more context and are more likely to contain a complete answer in a single retrieval hit. The trade-off is lower precision, because the chunk may also include unrelated material that distracts the model.

Semantic chunking groups sentences by meaning instead of by character count. It embeds each sentence and starts a new chunk when the similarity between neighboring sentences drops, so each chunk covers one coherent idea.

Chunk metadata should record the source document, the page or section, and the chunk's position in the original document. Good metadata makes citations possible and lets a developer trace a retrieved chunk back to its source.

A common mistake is to chunk before cleaning the text. Headers, footers, page numbers, and boilerplate should be removed first, because otherwise they appear in many chunks and pollute retrieval results.

There is no single best chunk size. The right choice depends on the documents, the questions, and the embedding model, so it should be measured with an evaluation harness on a golden dataset rather than guessed.
