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

class BagOfWord:    
    @staticmethod
    def predict_from_training_set(training_sets: list[tuple[str, bool]], test_documents: list[str]) -> list[bool]:
        from sklearn.feature_extraction.text import CountVectorizer
        bow_vectorizer = CountVectorizer()
        document_list = [training_set[0] for training_set in training_sets]
        merged_document = " ".join(document_list)
        vectors = bow_vectorizer.fit_transform([training_set[0] for training_set in training_sets])
        test_vectors = bow_vectorizer.transform(test_documents)
        labels = [training_set[1] for training_set in training_sets]
        from sklearn.naive_bayes import MultinomialNB
        classifier = MultinomialNB()
        classifier.fit(vectors, labels)
        predictions = classifier.predict(test_vectors)
        return predictions.tolist()