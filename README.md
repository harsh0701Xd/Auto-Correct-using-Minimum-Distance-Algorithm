# Auto-Correct-using-Minimum-Edit-Distance-Algorithm

## Table of Contents
1. [Overview](#overview)
2. [Introduction](#introduction)
3. [Dataset Used](#dataset-used)
4. [Data Preprocessing](#data-preprocessing)
5. [Algorithm Used](#algorithm-used)
6. [Minimum Edit Distance Algorithm](#minimum-edit-distance-algorithm)
7. [Final Result](#final-result)

## Overview
Autocorrect is a feature commonly used on cell phones and computers. This project delves into the mechanisms behind autocorrect. While the model implemented here is not identical to the one on your phone, it is still quite effective.

Through this project, one will learn how to:
- Obtain a word count from a corpus
- Determine word probability within the corpus
- Manipulate and filter strings
- Implement minimum edit distance to compare strings and find the optimal editing path
- Understand the principles of dynamic programming

Such systems are prevalent in various applications. For instance, if someone types “I am lern-ingg”, it is highly likely they intended to write “learning”.

## Introduction
This project involves implementing models that correct words that are one or two edit distances away. Two words are said to be `n` edit distances apart when `n` edits are required to transform one word into the other. An edit can consist of one of the following actions:
- Deletion (removing a letter): ‘hat’ => ‘at, ha, ht’
- Switching (swapping two adjacent letters): ‘eta’ => ‘eat, tea,…’
- Replacement (changing one letter to another): ‘jat’ => ‘hat, rat, cat, mat, …’
- Insertion (adding a letter): ‘te’ => ‘the, ten, ate, …’

These four methods will be used to implement an auto-correct feature. To achieve this, probabilities must be computed to determine the likelihood that a given word is correct.

The auto-correct model being implemented was originally created by Peter Norvig in 2007. His original article can serve as a useful reference.

The objective of the spell check model is to compute the following probability:
$$P(c|w) = \frac{P(w|c)\times P(c)}{P(w)} \tag{Eqn-1}$$
This equation is derived from Bayes’ Rule.

## Dataset Used
The dataset used for this project is a text file named `shakespeare.txt`. This file contains a large corpus of text from which word probabilities will be computed.

## Data Preprocessing
### process_data
The `process_data` function reads in a corpus (text file), converts all text to lowercase, and returns a list of words.

```python
def process_data(file_name):
    words = []
    with open(file_name) as f:
        lines = f.read()
        lines = lines.lower()
        words = re.findall('\w+', lines)
    return words
```

### get_count
The `get_count` function returns a dictionary where the keys are words and the values are the number of times each word appears in the corpus.

```python
def get_count(word_l):
    word_count_dict = {}
    for word in word_l:
        if word in word_count_dict:
            word_count_dict[word] += 1
        else:
            word_count_dict[word] = 1
    return word_count_dict
```

### get_probs
The `get_probs` function computes the probability that each word will appear if randomly selected from the corpus of words.

```python
def get_probs(word_count_dict):
    probs = {}
    total_count = sum(word_count_dict.values())
    for word, count in word_count_dict.items():
        probs[word] = count / total_count
    return probs
```

## Algorithm Used

### Edit Distance
This project involves implementing models that correct words that are one or two edit distances away. Two words are said to be n edit distances apart when n edits are required to transform one word into the other.

### String Manipulations
We will implement four functions to manipulate strings so that we can edit the erroneous strings and return the correct spellings:

delete_letter: Returns all possible strings with one character deleted.
switch_letter: Returns all possible strings with two adjacent letters switched.
replace_letter: Returns all possible strings with one character replaced by another.
insert_letter: Returns all possible strings with an additional character inserted.

```python
def delete_letter(word, verbose=False):
    delete_l = []
    split_l = [(word[:i], word[i:]) for i in range(len(word) + 1)]
    delete_l = [L + R[1:] for L, R in split_l if R]
    return delete_l
```

```python
def switch_letter(word, verbose=False):
    switch_l = []
    split_l = [(word[:i], word[i:]) for i in range(len(word) - 1)]
    switch_l = [L + R[1] + R[0] + R[2:] for L, R in split_l if len(R) > 1]
    return switch_l
```

```python
def replace_letter(word, verbose=False):
    letters = 'abcdefghijklmnopqrstuvwxyz'
    replace_l = []
    split_l = [(word[:i], word[i:]) for i in range(len(word))]
    replace_l = [L + c + R[1:] for L, R in split_l if R for c in letters if c != R[0]]
    return replace_l
```

```python
def insert_letter(word, verbose=False):
    letters = 'abcdefghijklmnopqrstuvwxyz'
    insert_l = []
    split_l = [(word[:i], word[i:]) for i in range(len(word) + 1)]
    insert_l = [L + c + R for L, R in split_l for c in letters]
    return insert_l
```

### Combining the Edits
The edit_one_letter function returns all the possible edits that are one edit away from a word.

```python
def edit_one_letter(word, allow_switches=True):
    edit_one_set = set()
    edit_one_set.update(delete_letter(word))
    if allow_switches:
        edit_one_set.update(switch_letter(word))
    edit_one_set.update(replace_letter(word))
    edit_one_set.update(insert_letter(word))
    return edit_one_set
```

### Minimum Edit Distance Algorithm
The Minimum Edit Distance algorithm calculates the minimum number of edits required to transform one word into another. This is often referred to as the Levenshtein distance. The allowed operations are insertion, deletion, and substitution.

#### Definition
For two strings \( X \) and \( Y \) and their lengths \( n \) and \( m \), respectively, the minimum edit distance \( d(X,Y) \) is computed as follows:

1. If either string is empty, the distance is the length of the other string.
2. If the last characters of \( X \) and \( Y \) are the same, the distance is the distance of the prefixes.
3. Otherwise, consider the minimum cost of the three operations (insertion, deletion, substitution) applied to the last character.

#### Implementation
The algorithm uses dynamic programming to store the results of subproblems in a matrix \( dp \), where \( dp[i][j] \) represents the edit distance between the first \( i \) characters of \( X \) and the first \( j \) characters of \( Y \).

```python
def min_edit_distance(source, target):
    n = len(source)
    m = len(target)
    dp = [[0 for j in range(m + 1)] for i in range(n + 1)]

    for i in range(n + 1):
        dp[i][0] = i

    for j in range(m + 1):
        dp[0][j] = j

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if source[i - 1] == target[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = min(dp[i - 1][j] + 1,  # Deletion
                               dp[i][j - 1] + 1,  # Insertion
                               dp[i - 1][j - 1] + 1)  # Substitution

    return dp[n][m]
```

The above function calculates the minimum edit distance between two words and can be used to evaluate the similarity between words in the context of the autocorrect system.

## Final Result
The final result of this project is an autocorrect system that can suggest corrections for misspelled words based on the implemented edit distance algorithms and calculated word probabilities.
