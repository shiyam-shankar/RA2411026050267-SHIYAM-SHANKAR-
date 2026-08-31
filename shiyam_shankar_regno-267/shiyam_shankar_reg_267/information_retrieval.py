"""
Information Retrieval System
Demonstrates TF, DF, IDF, TF-IDF, Cosine Similarity, and Range calculations.
Created as a college-level Python assignment.
"""

import os
import math
import re
import csv
from collections import Counter

# Small built-in set of English stop words
STOP_WORDS = {
    'a', 'about', 'above', 'after', 'again', 'against', 'all', 'am', 'an', 'and', 'any', 'are', 'arent',
    'as', 'at', 'be', 'because', 'been', 'before', 'being', 'below', 'between', 'both', 'but', 'by', 'cant',
    'cannot', 'could', 'couldnt', 'did', 'didnt', 'do', 'does', 'doesnt', 'doing', 'dont', 'down', 'during',
    'each', 'few', 'for', 'from', 'further', 'had', 'hadnt', 'has', 'hasnt', 'have', 'havent', 'having',
    'he', 'hed', 'hell', 'hes', 'her', 'here', 'heres', 'hers', 'herself', 'him', 'himself', 'his', 'how',
    'hows', 'i', 'id', 'ill', 'im', 'ive', 'if', 'in', 'into', 'is', 'isnt', 'it', 'its', 'itself',
    'lets', 'me', 'more', 'most', 'mustnt', 'my', 'myself', 'no', 'nor', 'not', 'of', 'off', 'on', 'once',
    'only', 'or', 'other', 'ought', 'our', 'ours', 'ourselves', 'out', 'over', 'own', 'same', 'shant',
    'she', 'shed', 'shell', 'shes', 'should', 'shouldnt', 'so', 'some', 'such', 'than', 'that', 'thats',
    'the', 'their', 'theirs', 'them', 'themselves', 'then', 'there', 'theres', 'these', 'they', 'theyd',
    'theyll', 'theyre', 'theyve', 'this', 'those', 'through', 'to', 'too', 'under', 'until', 'up', 'very',
    'was', 'wasnt', 'we', 'wed', 'well', 'were', 'weve', 'werent', 'what', 'whats', 'when', 'whens',
    'where', 'wheres', 'which', 'while', 'who', 'whos', 'whom', 'why', 'whys', 'with', 'wont', 'would',
    'wouldnt', 'you', 'youd', 'youll', 'youre', 'youve', 'your', 'yours', 'yourself', 'yourselves'
}

def load_documents(directory):
    """Loads all .txt documents from the specified directory."""
    documents = {}
    if not os.path.exists(directory):
        print(f"Error: Directory '{directory}' does not exist.")
        return documents
        
    for filename in sorted(os.listdir(directory)):
        if filename.endswith('.txt'):
            path = os.path.join(directory, filename)
            with open(path, 'r', encoding='utf-8') as f:
                documents[filename] = f.read()
    return documents

def preprocess_text(text):
    """
    Preprocesses the input text:
    1. Converts to lowercase.
    2. Removes punctuation using regex.
    3. Splits text into words/tokens.
    4. Filters out common stop words.
    """
    # Convert to lowercase
    text = text.lower()
    # Remove punctuation (keep alphanumeric characters and spaces)
    text = re.sub(r'[^a-z0-9\s]', '', text)
    # Split into words
    tokens = text.split()
    # Filter stop words
    filtered_tokens = [w for w in tokens if w not in STOP_WORDS]
    return filtered_tokens

def calculate_tf(tokens):
    """
    Calculates Term Frequency (TF) for a document.
    TF = Count of term / Total number of terms in document
    """
    if not tokens:
        return {}
    counts = Counter(tokens)
    total_words = len(tokens)
    tf = {term: count / total_words for term, count in counts.items()}
    return tf

def calculate_document_frequency(all_docs_tokens):
    """
    Calculates Document Frequency (DF) for all terms in the vocabulary.
    DF = Number of documents containing the term
    """
    df = Counter()
    for tokens in all_docs_tokens.values():
        unique_terms = set(tokens)
        for term in unique_terms:
            df[term] += 1
    return dict(df)

def calculate_idf(df, N):
    """
    Calculates Inverse Document Frequency (IDF) for all terms in vocabulary.
    IDF = log(N / DF)
    """
    idf = {}
    for term, count in df.items():
        # Prevent division-by-zero (count will always be >= 1 here since it's from df)
        idf[term] = math.log(N / count)
    return idf

def calculate_tfidf(tf, idf):
    """
    Calculates TF-IDF vector for a single document.
    TF-IDF = TF * IDF
    """
    tfidf = {}
    for term, tf_val in tf.items():
        # If term is not in our training vocabulary (e.g. query terms not in vocabulary), idf is 0
        tfidf[term] = tf_val * idf.get(term, 0.0)
    return tfidf

def calculate_cosine_similarity(vec_a, vec_b):
    """
    Calculates Cosine Similarity between vector A and vector B.
    Formula: (A . B) / (|A| * |B|)
    """
    # Find all unique keys
    all_keys = set(vec_a.keys()).union(set(vec_b.keys()))
    
    # Calculate dot product
    dot_product = sum(vec_a.get(k, 0.0) * vec_b.get(k, 0.0) for k in all_keys)
    
    # Calculate magnitudes
    magnitude_a = math.sqrt(sum(v ** 2 for v in vec_a.values()))
    magnitude_b = math.sqrt(sum(v ** 2 for v in vec_b.values()))
    
    # Handle zero magnitude
    if magnitude_a == 0.0 or magnitude_b == 0.0:
        return 0.0
        
    return dot_product / (magnitude_a * magnitude_b)

def calculate_range(scores):
    """Calculates range: Max Score - Min Score."""
    if not scores:
        return 0.0, 0.0, 0.0
    max_score = max(scores)
    min_score = min(scores)
    return max_score, min_score, max_score - min_score

def rank_documents(scores):
    """Ranks documents based on scores in descending order."""
    return sorted(scores.items(), key=lambda item: item[1], reverse=True)

def save_results(query, ranked_docs, range_stats, all_tf, idf, all_tfidf, vocabulary, outputs_dir):
    """Saves TF, IDF, TF-IDF, Cosine Similarity and Range stats into output files."""
    os.makedirs(outputs_dir, exist_ok=True)
    
    # Sort vocabulary for consistent column order
    sorted_vocab = sorted(list(vocabulary))
    doc_names = sorted(list(all_tf.keys()))
    
    # 1. Save TF results
    with open(os.path.join(outputs_dir, 'tf_results.csv'), 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['Term'] + doc_names)
        for term in sorted_vocab:
            row = [term] + [all_tf[doc].get(term, 0.0) for doc in doc_names]
            writer.writerow(row)
            
    # 2. Save IDF results
    with open(os.path.join(outputs_dir, 'idf_results.csv'), 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['Term', 'Document Frequency', 'IDF'])
        for term in sorted_vocab:
            # We calculate Document Frequency dynamically or retrieve from DF
            df_val = sum(1 for doc in doc_names if term in all_tf[doc])
            writer.writerow([term, df_val, idf.get(term, 0.0)])
            
    # 3. Save TF-IDF results
    with open(os.path.join(outputs_dir, 'tfidf_results.csv'), 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['Term'] + doc_names)
        for term in sorted_vocab:
            row = [term] + [all_tfidf[doc].get(term, 0.0) for doc in doc_names]
            writer.writerow(row)
            
    # 4. Save Cosine Similarity results
    with open(os.path.join(outputs_dir, 'cosine_similarity_results.csv'), 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['Rank', 'Dataset', 'Similarity'])
        for idx, (doc, score) in enumerate(ranked_docs, start=1):
            writer.writerow([idx, doc, f"{score:.4f}"])
            
    # 5. Save Range results
    max_sim, min_sim, range_val = range_stats
    best_doc = ranked_docs[0][0] if ranked_docs else "None"
    with open(os.path.join(outputs_dir, 'range_results.txt'), 'w', encoding='utf-8') as f:
        f.write(f"Query: {query}\n")
        f.write(f"Maximum Similarity : {max_sim:.4f}\n")
        f.write(f"Minimum Similarity : {min_sim:.4f}\n")
        f.write(f"Range              : {range_val:.4f}\n")
        f.write(f"Most Relevant Document: {best_doc}\n")

def main():
    print("============================================================")
    print("            INFORMATION RETRIEVAL SYSTEM")
    print("============================================================")
    
    # Paths
    base_dir = os.path.dirname(os.path.abspath(__file__))
    datasets_dir = os.path.join(base_dir, 'datasets')
    outputs_dir = os.path.join(base_dir, 'outputs')
    
    # 1. Load Documents
    raw_docs = load_documents(datasets_dir)
    N = len(raw_docs)
    
    if N == 0:
        print("Error: No documents loaded from 'datasets/' folder. Exiting.")
        return
        
    print(f"Total documents loaded: {N}\n")
    for name in raw_docs.keys():
        print(f" - {name}")
    print()
    
    # 2. Preprocess Documents
    doc_tokens = {name: preprocess_text(text) for name, text in raw_docs.items()}
    
    # Build vocabulary of all unique terms across all documents
    vocabulary = set()
    for tokens in doc_tokens.values():
        vocabulary.update(tokens)
        
    # 3. Calculate TF for all documents
    all_tf = {name: calculate_tf(tokens) for name, tokens in doc_tokens.items()}
    
    # 4. Calculate Document Frequency (DF)
    df = calculate_document_frequency(doc_tokens)
    
    # 5. Calculate IDF using document collection
    idf = calculate_idf(df, N)
    
    # 6. Calculate TF-IDF for all documents
    all_tfidf = {name: calculate_tfidf(tf_val, idf) for name, tf_val in all_tf.items()}
    
    # Loop for user query interaction
    while True:
        query = input("Enter your search query (or press Enter to exit): ").strip()
        if not query:
            print("Exiting search session.")
            break
            
        print("\nProcessing query...")
        
        # Preprocess query
        query_tokens = preprocess_text(query)
        
        if not query_tokens:
            print("Query contained no valid terms after preprocessing.")
            print("All similarities will be 0.0.\n")
            # Create a dummy zero vector query
            query_tfidf = {}
        else:
            # Calculate Term Frequency (TF) for query
            query_tf = calculate_tf(query_tokens)
            # Calculate TF-IDF for query using document collection's IDF values
            query_tfidf = calculate_tfidf(query_tf, idf)
            
        # Calculate Cosine Similarity with all documents
        similarities = {}
        for doc_name, doc_tfidf in all_tfidf.items():
            similarities[doc_name] = calculate_cosine_similarity(query_tfidf, doc_tfidf)
            
        # Rank documents
        ranked_docs = rank_documents(similarities)
        
        # Calculate Range Stats
        max_sim, min_sim, range_val = calculate_range(list(similarities.values()))
        range_stats = (max_sim, min_sim, range_val)
        
        # Display Results
        print("\n------------------------------------------------------------")
        print("COSINE SIMILARITY RESULTS")
        print("------------------------------------------------------------")
        print(f"{'Rank':<8}{'Document':<45}{'Score'}")
        print("-" * 60)
        for idx, (doc, score) in enumerate(ranked_docs, start=1):
            print(f"{idx:<8}{doc:<45}{score:.4f}")
            
        print("\n------------------------------------------------------------")
        print("MOST RELEVANT DOCUMENT")
        print("------------------------------------------------------------")
        best_doc, best_score = ranked_docs[0] if ranked_docs else ("None", 0.0)
        print(f"Document: {best_doc}")
        print(f"Similarity Score: {best_score:.4f}")
        print("-" * 60)
            
        # Save results to output folder (done silently)
        save_results(query, ranked_docs, range_stats, all_tf, idf, all_tfidf, vocabulary, outputs_dir)
        print()


if __name__ == '__main__':
    main()