"""
Bag of Words Feature Extractor

This script demonstrates the Bag of Words (BoW) model for feature extraction in Natural 
Language Processing (NLP). The BoW model represents documents as an unordered collection 
of words, discarding grammar and word order while preserving word frequency.

The script uses scikit-learn's CountVectorizer to:
  - Convert text documents into a numerical feature matrix
  - Count occurrences of each unique word across documents
  - Extract feature names (vocabulary) from the corpus

Applications:
  - Text classification and sentiment analysis
  - Topic modeling and document clustering
  - Information retrieval and search

Each row in the output matrix represents a document, and each column represents 
a unique word from the corpus. The cell values indicate word frequency within 
each document.
"""

from sklearn.feature_extraction.text import CountVectorizer


def main():
    """Execute the Bag of Words feature extraction workflow."""
    
    # ===== STEP 1: Create Sample Documents =====
    # These are the text documents we want to analyze. In real applications, this could be
    # product reviews, emails, social media posts, etc. We'll convert these sentences into
    # numbers that machine learning algorithms can understand.
    documents = [
        "I love programming in Python",
        "Python is a great language",
        "I enjoy machine learning"
    ]
    
    print("=" * 60)
    print("STEP 1: Original Documents")
    print("=" * 60)
    for i, doc in enumerate(documents, 1):
        print(f"  Document {i}: {doc}")
    
    
    # ===== STEP 2: Create a CountVectorizer =====
    # CountVectorizer is a tool from scikit-learn that converts text documents into
    # numerical data (a matrix). It counts how many times each unique word appears in
    # each document. This is called "vectorization" - turning words into vectors (lists of numbers).
    vectorizer = CountVectorizer()
    print("\n" + "=" * 60)
    print("STEP 2: Initialize CountVectorizer")
    print("=" * 60)
    print("  CountVectorizer will:")
    print("  - Find all unique words in the documents")
    print("  - Count how many times each word appears in each document")
    print("  - Create a matrix where rows = documents, columns = words")
    
    
    # ===== STEP 3: Fit and Transform =====
    # fit_transform() does two things:
    #   1. "fit" - LEARNS THE VOCABULARY (scans all documents to find unique words)
    #             NOTE: This is NOT "fitting a model" - it's just building a word dictionary
    #   2. "transform" - CONVERTS DOCUMENTS TO NUMBERS (creates the count matrix)
    # The result is stored in "bow_matrix", which is a sparse matrix (efficient for storing
    # mostly zeros - since most words don't appear in most documents)
    #
    # IMPORTANT: fit_transform() must be called on ALL training documents together.
    # You cannot fit on some documents and later add new ones without refitting everything.
    bow_matrix = vectorizer.fit_transform(documents)
    print("\n" + "=" * 60)
    print("STEP 3: Fit and Transform Documents")
    print("=" * 60)
    print(f"  Number of documents processed: {len(documents)}")
    print(f"  Total unique words found: {len(vectorizer.get_feature_names_out())}")
    print(f"  Matrix shape: (documents={bow_matrix.shape[0]}, unique_words={bow_matrix.shape[1]})")
    
    
    # ===== STEP 4: Extract Feature Names =====
    # Feature names are simply the unique words (vocabulary) that CountVectorizer discovered.
    # Each feature name corresponds to one column in our matrix. For example:
    # Feature 0 = "language", Feature 1 = "love", Feature 2 = "python", etc.
    feature_names = vectorizer.get_feature_names_out()
    print("\n" + "=" * 60)
    print("STEP 4: Extract Feature Names (Unique Words)")
    print("=" * 60)
    print("  These are all the unique words found in our documents:")
    print(f"  {', '.join(feature_names)}")
    
    
    # ===== STEP 5: Convert to Dense Array =====
    # CountVectorizer creates a "sparse matrix" (efficient but hard to read).
    # We convert it to a "dense array" (regular matrix that's easier to understand).
    # Each cell contains the count of how many times that word appears in that document.
    bow_array = bow_matrix.toarray()
    print("\n" + "=" * 60)
    print("STEP 5: Bag of Words Matrix (Word Counts)")
    print("=" * 60)
    print("  Each row = a document, Each column = a unique word")
    print("  Each cell = how many times that word appears in that document")
    print("\n  Word counts:\n")
    print(bow_array)
    
    
    # ===== STEP 6: Create a Readable Table =====
    # Let's display this in a more understandable format for students
    print("\n" + "=" * 60)
    print("STEP 6: Readable Format with Labels")
    print("=" * 60)
    print("\n  VOCABULARY (Column Names):")
    for idx, word in enumerate(feature_names):
        print(f"    Column {idx} = '{word}'")
    
    print("\n  WORD COUNTS BY DOCUMENT:")
    for doc_idx, counts in enumerate(bow_array, 1):
        print(f"\n    Document {doc_idx}: '{documents[doc_idx-1]}'")
        print(f"    Word counts: {dict(zip(feature_names, counts))}")
        print(f"    Total words: {counts.sum()}")


    # ===== INTERPRETATION EXAMPLE =====
    print("\n" + "=" * 60)
    print("EXAMPLE INTERPRETATION")
    print("=" * 60)
    print(f"""
  Looking at the first row (Document 1):
  - The word 'programming' appears {bow_array[0, list(feature_names).index('programming')]} time
  - The word 'python' appears {bow_array[0, list(feature_names).index('python')]} time
  - The word 'great' appears {bow_array[0, list(feature_names).index('great')]} time(s)
  
  This happens because Document 1 is: "{documents[0]}"
  It contains "Python" once and "programming" once, but doesn't contain the word "great".
    """)


if __name__ == "__main__":
    main()

# The current code is only doing feature extraction - it's NOT a predictive model.

# Current State:

# ✅ Converts text → numbers (Bag of Words vectors)
# ❌ Does NOT learn patterns or relationships
# ❌ Does NOT make predictions
# ❌ Does NOT classify sentiment or anything else
# What happens with new data?
# If you gave it a new document like "I love Python", CountVectorizer would:

# Need to be retrained on ALL documents (old + new)
# Create a completely new matrix
# Still just output numbers - no predictions
# To actually predict something, you'd need:

# BoW (feature extraction) → converts text to numbers
# Classifier (like Naive Bayes, SVM, etc.) → learns patterns from labeled data
# Prediction → classifies new documents
