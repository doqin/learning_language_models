"""
Learning natural language processing (NLP)
18/2/2026: Today's language model--Bag-of-words! (BoW)
* Anecdote:
So apparently there are many statistical language models
Bag-of-words is a statistical language model based on word count
Basically it doesn't give a shit about what the order or pattern, only count matters!
It's also a special case for an n-gram model, with n being 1. Whatever that means lol
For this example we're using Naive Bayes classifiers as it's great for spam filtering and document categorisation
The reason it's 'Naive' is cuz it assumes the features are independent of each others, meaning the presence of one feature does not affect the presence of another
Read more at: https://www.geeksforgeeks.org/machine-learning/naive-bayes-classifiers/
"""

from typing import TypeVar

T = TypeVar('T')
K = TypeVar('K')
V = TypeVar('V')

class BagOfWord:    
    @staticmethod
    def predict_from_training_set(training_sets: list[tuple[str, T]], test_documents: list[str]) -> list[T]:
        documents = [training_set[0] for training_set in training_sets]
        labels = [training_set[1] for training_set in training_sets]
        from sklearn.feature_extraction.text import CountVectorizer
        bow_vectorizer = CountVectorizer()
        vectors = bow_vectorizer.fit_transform(documents)
        test_vectors = bow_vectorizer.transform(test_documents)
        label_dictionary = BagOfWord.__create_label_dictionary(labels)
        index_dictionary = BagOfWord.__swap_dictionary_key_value(label_dictionary)
        index_labels = [label_dictionary[label] for label in labels]
        from sklearn.naive_bayes import MultinomialNB
        classifier = MultinomialNB()
        classifier.fit(vectors, index_labels)
        predictions = classifier.predict(test_vectors)
        return [index_dictionary[prediction] for prediction in predictions.tolist()]
    
    @staticmethod
    def __create_label_dictionary(labels: list[T]) -> dict[T, int]:
        label_dictionary = dict.fromkeys(labels, 0)
        index = 0
        for key in label_dictionary:
            label_dictionary[key] = index
            index += 1
        return label_dictionary
    
    @staticmethod
    def __swap_dictionary_key_value(dictionary: dict[K, V]) -> dict[V, K]:
        return {value: key for key, value in dictionary.items()}