
# ============================================================
# NATURAL LANGUAGE PROCESSING (NLP) - UNITS I TO V
# Dataset: 10 lines containing URLs, contractions and punctuation
# ============================================================

import re
import math
import random
from collections import Counter, defaultdict

import nltk

# Download NLTK resources needed for linguistic processing.
# The first run requires an internet connection.
nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)
nltk.download("averaged_perceptron_tagger_eng", quiet=True)
nltk.download("averaged_perceptron_tagger", quiet=True)
nltk.download("maxent_ne_chunker_tab", quiet=True)
nltk.download("words", quiet=True)
nltk.download("wordnet", quiet=True)
nltk.download("omw-1.4", quiet=True)

from nltk import word_tokenize, pos_tag
from nltk.corpus import wordnet
from nltk.chunk import ne_chunk
from nltk.tree import Tree

# ============================================================
# DATASET: EXACTLY 10 LINES
# ============================================================

dataset = [
    "Visit https://example.com today; it won't take long!",
    "Alice can't open the website (https://openai.com) : help her.",
    "The clever student reads books, but the lazy cat sleeps.",
    "Python is useful; Java is popular, and mango tastes sweet.",
    "John lives in Chennai and studies computer science.",
    "The small dog chased a ball, because it was running fast.",
    "https://nltk.org provides tools for language processing.",
    "She said, \"I don't know!\" (maybe the answer is correct).",
    "A bright bird flies above trees; music sounds beautiful.",
    "Google works in California, and the team won't stop learning."
]

# Combine the lines into one document.
text = "\n".join(dataset)

print("=" * 65)
print("NLP DATASET")
print("=" * 65)

for number, line in enumerate(dataset, start=1):
    print(f"{number}. {line}")


# ============================================================
# UNIT I: TEXT PREPROCESSING
# Tokenization, cleaning, stopwords, stemming and lemmatization
# ============================================================

print("\n\nUNIT I - TEXT PREPROCESSING")

# Identify URLs before removing punctuation.
urls = re.findall(r"https?://\S+", text)
print("\nURLs:", urls)

# Expand common contractions for demonstration.
# This simple dictionary does not cover every English contraction.
contractions = {
    "can't": "cannot",
    "won't": "will not",
    "don't": "do not",
    "isn't": "is not",
    "it's": "it is"
}

expanded_text = text
for contraction, expansion in contractions.items():
    expanded_text = re.sub(
        re.escape(contraction),
        expansion,
        expanded_text,
        flags=re.IGNORECASE
    )

print("\nText after contraction expansion:")
print(expanded_text[:250], "...")

# Tokenization: split the text into words and punctuation.
tokens = word_tokenize(text)
print("\nFirst 30 tokens:", tokens[:30])

# Convert words to lowercase.
lower_tokens = [token.lower() for token in tokens]

# Remove URLs and punctuation, keeping alphabetic words.
clean_tokens = [
    token for token in lower_tokens
    if token.isalpha() and not token.startswith("http")
]

print("\nClean tokens:", clean_tokens[:30])

# Remove common stopwords.
stop_words = set(nltk.corpus.stopwords.words("english")) if False else {
    "the", "a", "an", "is", "are", "and", "but",
    "it", "in", "for", "to", "of", "was", "her"
}

filtered_tokens = [
    word for word in clean_tokens if word not in stop_words
]

print("\nAfter stopword removal:", filtered_tokens[:30])

# Stemming: reduce words to approximate word roots.
from nltk.stem import PorterStemmer, WordNetLemmatizer

stemmer = PorterStemmer()
stemmed_words = [stemmer.stem(word) for word in filtered_tokens]

print("\nStemmed words:", stemmed_words[:25])

# Lemmatization: convert words to dictionary forms.
lemmatizer = WordNetLemmatizer()
lemmatized_words = [
    lemmatizer.lemmatize(word) for word in filtered_tokens
]

print("\nLemmatized words:", lemmatized_words[:25])

# Character-level punctuation analysis.
punctuation = re.findall(r"[^\w\s]", text)
print("\nPunctuation symbols:", Counter(punctuation))


# ============================================================
# UNIT I: N-GRAM LANGUAGE MODELS
# Unigram, bigram, trigram and next-word probability
# ============================================================

print("\n\nUNIT I - N-GRAM LANGUAGE MODELS")

# Create n-grams from the cleaned tokens.
def make_ngrams(words, n):
    return [
        tuple(words[i:i + n])
        for i in range(len(words) - n + 1)
    ]

unigrams = make_ngrams(filtered_tokens, 1)
bigrams = make_ngrams(filtered_tokens, 2)
trigrams = make_ngrams(filtered_tokens, 3)

print("Unigrams:", unigrams[:10])
print("Bigrams:", bigrams[:10])
print("Trigrams:", trigrams[:10])

# Estimate a bigram probability using maximum likelihood.
unigram_counts = Counter(filtered_tokens)
bigram_counts = Counter(bigrams)

def bigram_probability(previous_word, current_word):
    numerator = bigram_counts[(previous_word, current_word)]
    denominator = unigram_counts[previous_word]

    if denominator == 0:
        return 0.0

    return numerator / denominator

print(
    "P('student' | 'clever') =",
    bigram_probability("clever", "student")
)

# Generate the most common next word after a given word.
def predict_next_word(previous_word):
    candidates = {
        pair[1]: count
        for pair, count in bigram_counts.items()
        if pair[0] == previous_word
    }

    if not candidates:
        return "No prediction available"

    return max(candidates, key=candidates.get)

print("Next word after 'clever':", predict_next_word("clever"))


# ============================================================
# UNIT I: SMOOTHING TECHNIQUES
# Add-one / Laplace smoothing
# ============================================================

print("\nUNIT I - LAPLACE SMOOTHING")

vocabulary = set(filtered_tokens)
vocab_size = len(vocabulary)

def laplace_probability(previous_word, current_word):
    numerator = bigram_counts[(previous_word, current_word)] + 1
    denominator = unigram_counts[previous_word] + vocab_size

    return numerator / denominator

print(
    "Smoothed probability:",
    laplace_probability("clever", "student")
)

# Even an unseen bigram gets a nonzero probability
# when both words belong to the vocabulary.
print(
    "Unseen bigram probability:",
    laplace_probability("student", "beautiful")
)

# Perplexity for a sequence under a Laplace-smoothed bigram model.
def bigram_perplexity(words):
    if len(words) < 2:
        return float("inf")

    log_probability = 0.0

    for i in range(1, len(words)):
        probability = laplace_probability(
            words[i - 1], words[i]
        )
        log_probability += math.log(probability)

    return math.exp(-log_probability / (len(words) - 1))

print("Bigram perplexity:", bigram_perplexity(filtered_tokens))


# ============================================================
# UNIT I: BYTE PAIR ENCODING (BPE)
# Repeatedly merge the most frequent adjacent symbol pair
# ============================================================

print("\nUNIT I - BYTE PAIR ENCODING")

def simple_bpe(words, merge_steps=10):
    # Start with character-level symbols and an end-of-word marker.
    word_tokens = [
        list(word) + ["</w>"] for word in words
    ]
    learned_merges = []

    for _ in range(merge_steps):
        pair_counts = Counter()

        # Count adjacent pairs across all words.
        for symbols in word_tokens:
            for i in range(len(symbols) - 1):
                pair_counts[(symbols[i], symbols[i + 1])] += 1

        if not pair_counts:
            break

        most_common_pair, frequency = pair_counts.most_common(1)[0]

        if frequency < 2:
            break

        merged_symbol = "".join(most_common_pair)
        learned_merges.append((most_common_pair, frequency))

        # Merge occurrences of the selected pair.
        new_word_tokens = []

        for symbols in word_tokens:
            merged = []
            i = 0

            while i < len(symbols):
                if (
                    i < len(symbols) - 1
                    and (symbols[i], symbols[i + 1]) == most_common_pair
                ):
                    merged.append(merged_symbol)
                    i += 2
                else:
                    merged.append(symbols[i])
                    i += 1

            new_word_tokens.append(merged)

        word_tokens = new_word_tokens

    return word_tokens, learned_merges

bpe_tokens, bpe_merges = simple_bpe(filtered_tokens, 10)
print("BPE tokens:", bpe_tokens[:10])
print("Learned merges:", bpe_merges)

# This is a teaching implementation of BPE, not a production tokenizer.


# ============================================================
# UNIT I: NEURAL LANGUAGE MODEL
# A small feed-forward neural model predicting the next word
# ============================================================

print("\nUNIT I - NEURAL LANGUAGE MODEL")

try:
    import numpy as np
    import torch
    import torch.nn as nn

    torch.manual_seed(42)

    # Keep only a small vocabulary for a quick demonstration.
    model_words = filtered_tokens
    model_vocab = sorted(set(model_words))
    word_to_id = {word: i for i, word in enumerate(model_vocab)}
    id_to_word = {i: word for word, i in word_to_id.items()}

    # Use two preceding words to predict the next word.
    context_size = 2
    training_examples = []

    for i in range(context_size, len(model_words)):
        context = model_words[i - context_size:i]
        target = model_words[i]

        training_examples.append((
            [word_to_id[w] for w in context],
            word_to_id[target]
        ))

    class NeuralLanguageModel(nn.Module):
        def __init__(self, vocab_size, embedding_size=16, hidden_size=32):
            super().__init__()

            self.embedding = nn.Embedding(vocab_size, embedding_size)
            self.network = nn.Sequential(
                nn.Linear(context_size * embedding_size, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, vocab_size)
            )

        def forward(self, context_ids):
            embeddings = self.embedding(context_ids)
            flattened = embeddings.reshape(embeddings.size(0), -1)
            return self.network(flattened)

    neural_model = NeuralLanguageModel(len(model_vocab))
    optimizer = torch.optim.Adam(neural_model.parameters(), lr=0.01)
    loss_function = nn.CrossEntropyLoss()

    # Train for a few epochs using this small demonstration dataset.
    for epoch in range(20):
        total_loss = 0.0

        for context, target in training_examples:
            x = torch.tensor([context], dtype=torch.long)
            y = torch.tensor([target], dtype=torch.long)

            optimizer.zero_grad()
            prediction = neural_model(x)
            loss = loss_function(prediction, y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

    print("Neural model training loss:", round(total_loss, 4))

    # Predict a word from the last two dataset words.
    if len(model_words) >= 2:
        last_context = model_words[-2:]
        input_ids = torch.tensor(
            [[word_to_id[w] for w in last_context]],
            dtype=torch.long
        )

        with torch.no_grad():
            scores = neural_model(input_ids)
            predicted_id = scores.argmax(dim=1).item()

        print("Context:", last_context)
        print("Predicted next word:", id_to_word[predicted_id])

except ImportError:
    print("Install torch and numpy to run the neural language model.")


# ============================================================
# UNIT II: PART-OF-SPEECH (POS) TAGGING
# Identify nouns, verbs, adjectives, adverbs and other tags
# ============================================================

print("\n\nUNIT II - POS TAGGING")

sample_sentence = "The clever student reads books quickly."
sample_tokens = word_tokenize(sample_sentence)

# NLTK assigns a POS tag to each token.
pos_results = pos_tag(sample_tokens)

print("Sentence:", sample_sentence)
print("POS tags:", pos_results)

# Example tags: NN = noun, VB = verb, JJ = adjective, RB = adverb.

# Additional POS analysis: open-class vs closed-class words.
# Open-class words usually add new content; closed-class words express grammar.
open_class_tags = {"NN", "NNS", "NNP", "NNPS", "VB", "VBD", "VBG",
                   "VBN", "VBP", "VBZ", "JJ", "JJR", "JJS", "RB", "RBR", "RBS"}
closed_class_tags = {"DT", "IN", "PRP", "PRP$", "CC"}

print("\nOpen-class words (NN, VB, JJ, RB):")
print([(word, tag) for word, tag in pos_results if tag in open_class_tags])
print("Closed-class words (DT, IN, PRP, CC):")
print([(word, tag) for word, tag in pos_results if tag in closed_class_tags])



# ============================================================
# UNIT II: CONTEXT-FREE GRAMMAR (CFG)
# Build a grammar and parse a sentence
# ============================================================

print("\nUNIT II - CFG PARSING")

from nltk import CFG, ChartParser

grammar = CFG.fromstring("""
    S -> NP VP
    NP -> Det N | Det Adj N
    VP -> V NP | V
    Det -> 'the' | 'a'
    Adj -> 'clever' | 'small'
    N -> 'student' | 'dog' | 'ball' | 'book'
    V -> 'reads' | 'chased' | 'sleeps'
""")

parser = ChartParser(grammar)
cfg_sentence = ["the", "clever", "student", "reads", "a", "book"]

print("Sentence:", " ".join(cfg_sentence))

try:
    for tree in parser.parse(cfg_sentence):
        print(tree)
        tree.pretty_print()
except ValueError:
    print("Sentence contains a word not covered by the grammar.")


# ============================================================
# UNIT II: PCFG
# Probabilistic Context-Free Grammar
# ============================================================

print("\nUNIT II - PCFG")

from nltk import PCFG, ViterbiParser

pcfg_grammar = PCFG.fromstring("""
    S -> NP VP [1.0]
    NP -> Det N [0.6] | Det Adj N [0.4]
    VP -> V NP [0.7] | V [0.3]
    Det -> 'the' [0.5] | 'a' [0.5]
    Adj -> 'clever' [0.5] | 'small' [0.5]
    N -> 'student' [0.4] | 'book' [0.3] | 'dog' [0.3]
    V -> 'reads' [0.5] | 'sleeps' [0.5]
""")

pcfg_parser = ViterbiParser(pcfg_grammar)
pcfg_sentence = ["the", "student", "reads", "a", "book"]

try:
    for tree in pcfg_parser.parse(pcfg_sentence):
        print(tree)
        print("Parse probability:", tree.prob())
except ValueError:
    print("No PCFG parse found.")


# ============================================================
# UNIT II: DEPENDENCY PARSING
# Demonstrate head-dependent relations with spaCy
# ============================================================

print("\nUNIT II - DEPENDENCY PARSING")

try:
    import spacy

    # Install the model separately:
    # python -m spacy download en_core_web_sm
    nlp = spacy.load("en_core_web_sm")
    doc = nlp("The clever student reads books quickly.")

    for token in doc:
        print(
            token.text,
            "| POS:", token.pos_,
            "| Head:", token.head.text,
            "| Dependency:", token.dep_

        )

    # Draw the dependency tree in a Jupyter notebook.
    # In a notebook, this displays a labeled dependency diagram.
    from spacy import displacy
    displacy.render(doc, style="dep", jupyter=True, options={"compact": True})

except (ImportError, OSError):
    print("Optional: pip install spacy")
    print("Then run: python -m spacy download en_core_web_sm")
    print("Dependency parsing requires the English model.")


# ============================================================
# UNIT III: WORDNET
# Synonyms, definitions and semantic relationships
# ============================================================

print("\n\nUNIT III - WORDNET")

word = "student"
synsets = wordnet.synsets(word)

for synset in synsets[:5]:
    print("Synset:", synset.name())
    print("Definition:", synset.definition())
    print("Examples:", synset.examples())

    # Print synonyms from the first few synsets.
    synonyms = {
        lemma.name()
        for lemma in synset.lemmas()
    }
    print("Synonyms:", synonyms)

# Semantic similarity using WordNet.
dog_synset = wordnet.synset("dog.n.01")
cat_synset = wordnet.synset("cat.n.01")

print(
    "Dog-cat similarity:",
    dog_synset.path_similarity(cat_synset)
)


# ============================================================
# UNIT III: VECTOR SPACE MODEL
# Represent documents as numerical vectors using TF-IDF
# ============================================================

print("\nUNIT III - VECTOR SPACE MODEL")

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

documents = [
    "Python is useful for programming",
    "Java is useful for software development",
    "Cats and dogs are animals",
    "Python helps students learn programming"
]

vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(documents)

print("Vocabulary:", vectorizer.get_feature_names_out())
print("TF-IDF matrix shape:", tfidf_matrix.shape)
print("TF-IDF matrix:\n", tfidf_matrix.toarray())

# Compare documents using cosine similarity.
similarity_matrix = cosine_similarity(tfidf_matrix)

print("\nDocument similarity matrix:")
print(similarity_matrix)

# Top-k document retrieval using TF-IDF and cosine similarity.
query = "Python programming for students"
query_vector = vectorizer.transform([query])
query_scores = cosine_similarity(query_vector, tfidf_matrix).flatten()
top_k = 3
top_indices = query_scores.argsort()[::-1][:top_k]

print("\nQuery:", query)
print(f"Top-{top_k} most similar documents:")
for rank, index in enumerate(top_indices, start=1):
    print(f"{rank}. Score={query_scores[index]:.3f} | {documents[index]}")


print(
    "Similarity between documents 1 and 4:",
    round(similarity_matrix[0, 3], 3)
)


# ============================================================
# UNIT III: WORD EMBEDDINGS
# Train a small Word2Vec model when gensim is available
# ============================================================

print("\nUNIT III - WORD EMBEDDINGS")

try:
    from gensim.models import Word2Vec

    embedding_sentences = [
        word_tokenize(line.lower())
        for line in dataset
    ]

    # Small demonstration embeddings trained on our 10 lines.
    word2vec_model = Word2Vec(
        sentences=embedding_sentences,
        vector_size=50,
        window=3,
        min_count=1,
        workers=1,
        epochs=100,
        seed=42
    )

    print("Embedding for 'student':")
    print(word2vec_model.wv["student"])

    print("Words similar to 'student':")
    print(word2vec_model.wv.most_similar("student", topn=3))

except ImportError:
    print("Install gensim to train Word2Vec: pip install gensim")

# PPMI (Positive Pointwise Mutual Information) with a context window of +/- 2.
# Counts each target word with words up to two positions to its left/right.
print("\nPPMI word-context matrix (window +/- 2)")
ppmi_sentences = [
    [token.lower() for token in word_tokenize(line) if token.isalpha()]
    for line in dataset
]
pair_counts = Counter()
target_counts = Counter()
context_counts = Counter()
total_pairs = 0
window_size = 2

for sentence in ppmi_sentences:
    for i, target_word in enumerate(sentence):
        left = max(0, i - window_size)
        right = min(len(sentence), i + window_size + 1)
        for j in range(left, right):
            if i == j:
                continue
            context_word = sentence[j]
            pair_counts[(target_word, context_word)] += 1
            target_counts[target_word] += 1
            context_counts[context_word] += 1
            total_pairs += 1

ppmi = {}
for (target_word, context_word), count in pair_counts.items():
    pmi = math.log2((count * total_pairs) /
                    (target_counts[target_word] * context_counts[context_word]))
    ppmi[(target_word, context_word)] = max(pmi, 0.0)

print("Sample PPMI values:", list(ppmi.items())[:10])

# Explicit Word2Vec skip-gram model (sg=1), with reproducibility settings.
try:
    from gensim.models import Word2Vec
    skipgram_model = Word2Vec(
        sentences=embedding_sentences,
        vector_size=50,
        window=2,
        min_count=1,
        sg=1,          # 1 = skip-gram; 0 = CBOW
        seed=42,
        workers=1,     # one worker helps reproducibility
        epochs=100
    )
    print("Skip-gram embedding for 'student':")
    if "student" in skipgram_model.wv:
        print(skipgram_model.wv["student"])
except (ImportError, NameError) as error:
    print("Skip-gram demonstration unavailable:", error)



# ============================================================
# UNIT III: TEXT CLASSIFICATION
# TF-IDF + Naive Bayes and performance evaluation
# ============================================================

print("\nUNIT III - TEXT CLASSIFICATION")

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

# A tiny labelled dataset for demonstrating supervised learning.
# More labelled examples are required for meaningful evaluation.
training_texts = [
    "Python code and software development",
    "Java programming and computer applications",
    "The football team won the match",
    "The player scored a goal in football",
    "Machine learning models process data",
    "The team practises football every day",
    "Artificial intelligence learns from data",
    "The football coach trained the players",
    "Computer programming uses algorithms",
    "The team scored another goal"
]

training_labels = [
    "technology", "technology", "sports", "sports",
    "technology", "sports", "technology", "sports",
    "technology", "sports"
]

X_train, X_test, y_train, y_test = train_test_split(
    training_texts,
    training_labels,
    test_size=0.3,
    random_state=42,
    stratify=training_labels
)

classifier = Pipeline([
    ("tfidf", TfidfVectorizer()),
    ("model", MultinomialNB())
])

classifier.fit(X_train, y_train)
predictions = classifier.predict(X_test)

print("Actual labels:", list(y_test))
print("Predictions:", list(predictions))
print("Accuracy:", accuracy_score(y_test, predictions))
print(classification_report(y_test, predictions, zero_division=0))

print(
    "New sentence classification:",
    classifier.predict(["Python programming and artificial intelligence"])
)


# ============================================================
# UNIT IV: COREFERENCE RESOLUTION
# Identify mentions that may refer to the same entity
# ============================================================

print("\n\nUNIT IV - COREFERENCE RESOLUTION")

coreference_text = (
    "Alice visited the library. She borrowed a book. "
    "The student returned it the next day."
)

print("Text:", coreference_text)

# Simple rule-based illustration, NOT a full coreference resolver.
# Real systems need contextual models and mention-resolution logic.
if re.search(r"\bAlice\b", coreference_text) and re.search(
    r"\bShe\b", coreference_text
):
    print("Possible coreference: 'She' refers to 'Alice'.")

if re.search(r"\ba book\b", coreference_text) and re.search(
    r"\bit\b", coreference_text
):
    print("Possible coreference: 'it' refers to 'a book'.")


# Advanced rule-based mention extraction and mention-pair scoring with spaCy.
# This is an educational heuristic, not a full coreference-resolution model.
print("\nUNIT IV - SPACY MENTIONS AND COREFERENCE CHAINS")
try:
    import spacy
    coref_nlp = spacy.load("en_core_web_sm")
    coref_doc = coref_nlp(
        "Mr. John Smith met Alice. He thanked her. "
        "Mrs. Brown said she knew John. The student lost his book."
    )

    # Hard-coded gender/number hints for a small demonstration.
    name_gender = {
        "john": "male", "alice": "female", "brown": "female",
        "mr.": "male", "mrs.": "female", "mr": "male", "mrs": "female"
    }
    pronoun_gender = {
        "he": "male", "him": "male", "his": "male",
        "she": "female", "her": "female", "hers": "female",
        "they": "unknown", "them": "unknown", "their": "unknown",
        "it": "neuter", "its": "neuter"
    }
    pronouns = set(pronoun_gender)
    mentions = []

    # Named entities (person names), noun chunks, and pronouns.
    for ent in coref_doc.ents:
        if ent.label_ == "PERSON":
            mentions.append({
                "text": ent.text, "start": ent.start, "end": ent.end,
                "sentence": ent.sent.start, "kind": "name",
                "head": ent.root.lemma_.lower(),
                "gender": next((name_gender[w.lower()] for w in ent.text.split()
                                if w.lower() in name_gender), "unknown"),
                "number": "singular"
            })

    for chunk in coref_doc.noun_chunks:
        if not any(m["start"] == chunk.start and m["end"] == chunk.end for m in mentions):
            mentions.append({
                "text": chunk.text, "start": chunk.start, "end": chunk.end,
                "sentence": chunk.sent.start, "kind": "noun_chunk",
                "head": chunk.root.lemma_.lower(),
                "gender": name_gender.get(chunk.root.text.lower(), "unknown"),
                "number": "plural" if chunk.root.tag_ in {"NNS", "NNPS"} else "singular"
            })

    for token in coref_doc:
        if token.lower_ in pronouns and token.pos_ == "PRON":
            mentions.append({
                "text": token.text, "start": token.i, "end": token.i + 1,
                "sentence": token.sent.start, "kind": "pronoun",
                "head": token.lemma_.lower(),
                "gender": pronoun_gender[token.lower_],
                "number": "plural" if token.tag_ in {"PRP", "PRP$"} and token.lower_ in {"they", "them", "their"} else "singular"
            })

    mentions.sort(key=lambda m: (m["start"], m["end"]))
    print("Mentions:")
    for mention in mentions:
        print(mention["text"], "|", mention["kind"], "| head:", mention["head"],
              "| gender:", mention["gender"], "| number:", mention["number"])

    def mention_pair_score(mention, candidate):
        # Higher score means the candidate is a more plausible antecedent.
        score = 0
        if mention["gender"] == "unknown" or candidate["gender"] == "unknown":
            score += 0
        elif mention["gender"] == candidate["gender"]:
            score += 2
        else:
            score -= 2

        if mention["number"] == candidate["number"]:
            score += 1
        else:
            score -= 1

        sentence_distance = abs(mention["sentence"] - candidate["sentence"])
        score += max(0, 3 - sentence_distance)  # nearer sentences score higher

        if mention["head"] == candidate["head"]:
            score += 2
        return score

    # Build simple chains by linking pronouns to the highest-scoring earlier mention.
    chains = []
    mention_to_chain = {}
    for index, mention in enumerate(mentions):
        if mention["kind"] == "pronoun":
            candidates = [m for m in mentions[:index]
                          if m["kind"] in {"name", "noun_chunk"}]
            if candidates:
                best = max(candidates, key=lambda candidate: mention_pair_score(mention, candidate))
                best_index = mentions.index(best)
                chain_id = mention_to_chain.get(best_index, len(chains))
                if chain_id == len(chains):
                    chains.append([best["text"]])
                    mention_to_chain[best_index] = chain_id
                chains[chain_id].append(mention["text"])
                mention_to_chain[index] = chain_id

    print("\nPossible coreference chains (heuristic):")
    for chain_number, chain in enumerate(chains, start=1):
        print(f"Chain {chain_number}:", " -> ".join(chain))

except (ImportError, OSError) as error:
    print("spaCy mention analysis unavailable:", error)
    print("Install spaCy and its English model:")
    print("pip install spacy")
    print("python -m spacy download en_core_web_sm")


# ============================================================
# UNIT IV: RNN-BASED MODELS
# A simple recurrent neural network for sequence processing
# ============================================================

print("\nUNIT IV - RNN")

try:
    import torch
    import torch.nn as nn

    class SimpleRNN(nn.Module):
        def __init__(self, input_size=10, hidden_size=16, output_size=2):
            super().__init__()

            self.rnn = nn.RNN(
                input_size=input_size,
                hidden_size=hidden_size,
                batch_first=True
            )
            self.output = nn.Linear(hidden_size, output_size)

        def forward(self, x):
            sequence_output, hidden_state = self.rnn(x)
            # Use the final time-step output for classification.
            last_output = sequence_output[:, -1, :]
            return self.output(last_output)

    rnn_model = SimpleRNN()

    # Batch of 4 sequences, each with 5 steps and 10 features.
    sample_sequence = torch.randn(4, 5, 10)
    rnn_output = rnn_model(sample_sequence)

    print("Input shape:", sample_sequence.shape)
    print("RNN output shape:", rnn_output.shape)
    print("Output logits:\n", rnn_output)

except ImportError:
    print("Install torch to run the RNN demonstration.")


# ============================================================
# UNIT IV: MACHINE TRANSLATION
# Simple word-dictionary translation example
# ============================================================

print("\nUNIT IV - MACHINE TRANSLATION")

# A toy dictionary demonstrates the basic idea only.
# It is not a complete English-to-Tamil translation system.
translation_dictionary = {
    "hello": "வணக்கம்",
    "student": "மாணவர்",
    "book": "புத்தகம்",
    "good": "நல்ல",
    "morning": "காலை",
    "thank": "நன்றி"
}

def translate_words(sentence):
    words = word_tokenize(sentence.lower())
    return " ".join(
        translation_dictionary.get(word, f"[{word}]")
        for word in words
        if word.isalpha()
    )

print("Translation:", translate_words("hello student"))


# ============================================================
# UNIT IV: MACHINE TRANSLATION EVALUATION
# BLEU score compares a candidate with a reference translation
# ============================================================

print("\nUNIT IV - BLEU EVALUATION")

from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

reference = [["the", "student", "reads", "a", "book"]]
candidate = ["the", "student", "reads", "book"]

bleu_score = sentence_bleu(
    reference,
    candidate,
    smoothing_function=SmoothingFunction().method1
)

print("Reference:", reference[0])
print("Candidate:", candidate)
print("BLEU score:", round(bleu_score, 4))


# ============================================================
# UNIT V: ATTENTION MECHANISM
# Scaled dot-product attention
# ============================================================

print("\n\nUNIT V - ATTENTION MECHANISM")

try:
    import torch
    import torch.nn.functional as F

    # Query, key and value matrices.
    query = torch.randn(1, 3, 8)
    key = torch.randn(1, 3, 8)
    value = torch.randn(1, 3, 8)

    # Attention(Q,K,V) = softmax(QK^T / sqrt(d_k)) V
    scores = torch.matmul(query, key.transpose(-2, -1))
    scores = scores / math.sqrt(query.size(-1))

    attention_weights = F.softmax(scores, dim=-1)
    attention_output = torch.matmul(attention_weights, value)

    print("Attention weights:\n", attention_weights)
    print("Attention output shape:", attention_output.shape)

except ImportError:
    print("Install torch to run the attention demonstration.")


# ============================================================
# UNIT V: TRANSFORMER ENCODER
# Self-attention and feed-forward processing
# ============================================================

print("\nUNIT V - TRANSFORMER")

try:
    import torch
    import torch.nn as nn

    # A small Transformer encoder with randomly initialized weights.
    encoder_layer = nn.TransformerEncoderLayer(
        d_model=32,
        nhead=4,
        dim_feedforward=64,
        batch_first=True
    )

    transformer_encoder = nn.TransformerEncoder(
        encoder_layer,
        num_layers=2
    )

    # Batch size 2, sequence length 5, embedding size 32.
    input_embeddings = torch.randn(2, 5, 32)
    transformer_output = transformer_encoder(input_embeddings)

    print("Input shape:", input_embeddings.shape)
    print("Transformer output shape:", transformer_output.shape)

except ImportError:
    print("Install torch to run the Transformer demonstration.")


# ============================================================
# UNIT V: BERT
# Use a pretrained Transformer model for contextual embeddings
# ============================================================

print("\nUNIT V - BERT")

try:
    from transformers import AutoTokenizer, AutoModel
    import torch

    # The first run downloads the model and requires internet access.
    bert_name = "google-bert/bert-base-uncased"

    bert_tokenizer = AutoTokenizer.from_pretrained(bert_name)
    bert_model = AutoModel.from_pretrained(bert_name)

    bert_sentence = "The student reads a useful book."
    bert_inputs = bert_tokenizer(
        bert_sentence,
        return_tensors="pt"
    )

    with torch.no_grad():
        bert_outputs = bert_model(**bert_inputs)

    print("BERT input tokens:",
          bert_tokenizer.convert_ids_to_tokens(bert_inputs["input_ids"][0]))
    print("BERT contextual embedding shape:",
          bert_outputs.last_hidden_state.shape)

except Exception as error:
    print("BERT demonstration skipped:", error)
    print("Install transformers and torch; check internet access for model download.")


# ============================================================
# UNIT V: NAMED ENTITY RECOGNITION (NER)
# Detect names of people, organizations and locations
# ============================================================

print("\nUNIT V - NAMED ENTITY RECOGNITION")

ner_sentence = (
    "Google works in California, and Alice studies in Chennai."
)

# First try spaCy's pretrained English model.
try:
    import spacy

    ner_nlp = spacy.load("en_core_web_sm")
    ner_doc = ner_nlp(ner_sentence)

    for entity in ner_doc.ents:
        print(
            "Entity:", entity.text,
            "| Label:", entity.label_
        )

except (ImportError, OSError):

    # Alternative: use a pretrained Hugging Face NER model.
    try:
        from transformers import pipeline

        ner_pipeline = pipeline(
            "ner",
            model="dslim/bert-base-NER",
            aggregation_strategy="simple"
        )

        for entity in ner_pipeline(ner_sentence):
            print(
                "Entity:", entity["word"],
                "| Label:", entity["entity_group"],
                "| Score:", round(entity["score"], 3)
            )

    except Exception as error:
        print("NER model unavailable:", error)
        print("Install spaCy or transformers to run pretrained NER.")


# ============================================================
# END OF ALL FIVE NLP UNITS
# ============================================================

print("\n" + "=" * 65)
print("ALL FIVE NLP UNITS COMPLETED")
print("=" * 65)
