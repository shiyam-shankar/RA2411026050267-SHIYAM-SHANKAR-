# shiyam-shankar-Reg267
============================================================
           INFORMATION RETRIEVAL SYSTEM ASSIGNMENT
============================================================

1. PROJECT TITLE
----------------
Python-Based Information Retrieval System

2. OBJECTIVE
------------
To build a simple, clean, and educational Information Retrieval system in Python.
The system calculates Term Frequency (TF), Inverse Document Frequency (IDF),
TF-IDF, and Cosine Similarity to search and rank 10 text-only documents 
against a user query. All calculations are implemented manually without 
advanced machine learning libraries (such as scikit-learn).

3. DATASET DESCRIPTION
----------------------
Exactly 10 text documents are located in the 'datasets/' directory.
Each file contains educational/reference material (approx. 300-500 words) 
covering the following topics:
1. Artificial Intelligence
2. Machine Learning
3. Deep Learning
4. Natural Language Processing
5. Computer Networks
6. Cybersecurity
7. Cloud Computing
8. Database Management Systems
9. Data Science
10. Software Engineering

The texts have natural overlapping vocabulary representing actual student essays.

4. TECHNOLOGIES USED
--------------------
- Programming Language: Python 3
- Libraries: os, math, re, csv, collections (all built-in standard libraries)

5. FORMULAS USED
----------------

* Term Frequency (TF):
  TF(t, d) = (Number of occurrences of term t in document d) / (Total number of terms in document d)

* Document Frequency (DF):
  DF(t) = Number of documents containing term t

* Inverse Document Frequency (IDF):
  IDF(t) = log(N / DF(t))
  where N = 10 (total number of documents)

* TF-IDF:
  TF-IDF(t, d) = TF(t, d) * IDF(t)

* Cosine Similarity:
  Cosine(A, B) = (A . B) / (|A| * |B|)
  where:
    A . B = dot product of vector A and vector B
    |A| = magnitude of vector A
    |B| = magnitude of vector B

* Range:
  Range = Maximum Similarity - Minimum Similarity

6. HOW TO RUN THE PROGRAM
-------------------------
1. Open a terminal or command prompt.
2. Navigate to the project directory.
3. Run the following command:
   python information_retrieval.py
4. Enter a search query when prompted (e.g. "machine learning").
5. The ranked results, similarity scores, range, and step-by-step samples will be displayed.
6. The detailed tables and metrics will be saved in the 'outputs/' folder as CSV and TXT files.

7. EXAMPLE QUERY
----------------
Query: machine learning algorithms

8. EXAMPLE OUTPUT
-----------------
------------------------------------------------------------
COSINE SIMILARITY RESULTS
------------------------------------------------------------
Rank    Document                                     Score
------------------------------------------------------------
1       dataset_02_machine_learning.txt              0.1420
2       dataset_03_deep_learning.txt                 0.0850
3       dataset_09_data_science.txt                  0.0540
...

------------------------------------------------------------
RANGE
------------------------------------------------------------
Maximum Similarity : 0.1420
Minimum Similarity : 0.0000
Range              : 0.1420

9. PROJECT STRUCTURE
--------------------
Information_Retrieval_Assignment/
│
├── datasets/
│   ├── dataset_01_artificial_intelligence.txt
│   ├── dataset_02_machine_learning.txt
│   ├── dataset_03_deep_learning.txt
│   ├── dataset_04_natural_language_processing.txt
│   ├── dataset_05_computer_networks.txt
│   ├── dataset_06_cybersecurity.txt
│   ├── dataset_07_cloud_computing.txt
│   ├── dataset_08_database_management.txt
│   ├── dataset_09_data_science.txt
│   └── dataset_10_software_engineering.txt
│
├── outputs/
│   ├── tf_results.csv
│   ├── idf_results.csv
│   ├── tfidf_results.csv
│   ├── cosine_similarity_results.csv
│   └── range_results.txt
│
├── information_retrieval.py
└── README.txt
