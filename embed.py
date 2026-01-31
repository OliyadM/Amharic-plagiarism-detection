# embed.py
# Sentence embedding using Gensim FastText model

import numpy as np
from preprocess import normalize_amharic, simple_sentence_split

# Use the fine-tuned Gensim model
MODEL_PATH = "fine_tuned_news_gensim.model"

def get_sentence_embedding(
    text: str,
    model,  # Gensim FastText model (no type hint needed)
    normalize: bool = True,
    min_words: int = 1
) -> np.ndarray | None:
    """
    Compute mean-pooled embedding for a single sentence using Gensim model.
    Returns 300-dim numpy vector or None if invalid/empty.
    """
    if not text or not text.strip():
        return None

    # Normalize if requested
    if normalize:
        text = normalize_amharic(text)

    # Split into words
    words = text.split()

    if len(words) < min_words:
        return None

    # Collect vectors (Gensim .wv interface)
    vectors = []
    for word in words:
        if word:
            try:
                vec = model.wv[word]
            except KeyError:
                vec = np.zeros(model.vector_size)  # zero for OOV
            vectors.append(vec)

    if not vectors:
        return None

    # Mean pool
    sentence_vec = np.mean(vectors, axis=0)

    return sentence_vec


if __name__ == "__main__":
    print("Loading Gensim model...")
    from gensim.models import FastText
    model = FastText.load(MODEL_PATH)
    print("Model loaded. Vector size:", model.vector_size)

    test_sentences = [
        "ተማሪው ትምህርት ቤት ሄደ።",
        "ተማሪዎቹ ትምህርት ቤት ተገኙ",
        "እኔ መጽሐፍ አነበብኩ እና ተደሰትኩ",
        "   ",  # empty
        "ሰላም ለሁሉም ሰው"
    ]

    for sent in test_sentences:
        emb = get_sentence_embedding(sent, model)
        if emb is not None:
            print(f"\nSentence: {sent}")
            print(f"Embedding shape: {emb.shape}")
            print(f"First 8 values: {emb[:8].round(4)}")
            print(f"Vector norm: {np.linalg.norm(emb):.4f}")
        else:
            print(f"\nSentence: {sent} → Skipped")