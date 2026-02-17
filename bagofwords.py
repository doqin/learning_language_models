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

import json
import io

class BagOfWord:    
    @staticmethod
    def predict_from_training_set(training_sets: list[tuple[str, bool]], test_document: str) -> bool:
        document_list = [training_set[0] for training_set in training_sets]
        merged_document = " ".join(document_list)
        features_dictionary, _ = BagOfWord.create_features_dictionary_from_document(merged_document)
        vectors = [BagOfWord.extract_features_from_document(document=training_set[0], features_dictionary=features_dictionary)[0] for training_set in training_sets]
        labels = [training_set[1] for training_set in training_sets]
        test_vector, _ = BagOfWord.extract_features_from_document(document=test_document, features_dictionary=features_dictionary)
        from sklearn.naive_bayes import MultinomialNB
        import numpy
        classifier = MultinomialNB()
        classifier.fit(vectors, labels)
        predictions = classifier.predict(numpy.array([test_vector]))
        return predictions[0]

    @staticmethod
    def create_features_dictionary_from_document(document: str) -> tuple[dict[str, int], list[str]]:
        """
        Creates a feature dictionary, which is a dictionary that stores the words present in the document as keys and an incrementing indices as values (used for creating feature vectors)
        
        :param document: string of the document content
        :type document: str
        :return features_dictionary: A dictionary with unique words present in the document as keys and the indices as values
        :return tokens: The tokens present in the document
        """
        from nltk.tokenize import word_tokenize # Turns string document into word tokens
        tokens = word_tokenize(text=document)
        return BagOfWord.create_features_dictionary_from_tokens(tokens), tokens

    @staticmethod
    def create_features_dictionary_from_tokens(document_tokens: list[str]) -> dict[str, int]:
        """
        Creates a feature dictionary, which is a dictionary that stores the words present in the document as keys and an incrementing indices as values (used for creating feature vectors)

        :param document_tokens: A list of tokens present in the input document
        :type document_tokens: list[str]
        :return: A dictionary with words present in the document as keys and the indices as values
        """
        features_dictionary: dict[str, int] = {}
        index = 0
        for token in document_tokens:
            if token not in features_dictionary:
                features_dictionary[token] = index
                index += 1
        return features_dictionary

    @staticmethod
    def extract_features_from_document(
        document: str,
        features_dictionary: dict[str, int]
    ) -> tuple[list[int], list[str]]:
        """
        Creates a feature vector (which is like a sheet for counting the presence of features in an item, here the features are the words present in a given document, and the items are the reference training documents). This is also known as feature extraction/vectorization

        :param document_tokens:  A list of tokens present in the input document
        :param features_dictionary: A dictionary of the unique words present in a reference document 
        """
        from nltk.tokenize import word_tokenize # Turns string document into word tokens
        tokens = word_tokenize(text=document)
        return BagOfWord.extract_features_from_tokens(document_tokens=tokens,features_dictionary=features_dictionary), tokens

    @staticmethod
    def extract_features_from_tokens(
        document_tokens: list[str], 
        features_dictionary: dict[str, int]
    ) -> list[int]:
        """
        Creates a feature vector (which is like a sheet for counting the presence of features in an item, here the features are the words present in a given document, and the items are the reference training documents). This is also known as feature extraction/vectorization

        :param document_tokens:  A list of tokens present in the input document
        :param features_dictionary: A dictionary of the unique words present in a reference document 
        """
        bow_vector = [0] * len(features_dictionary)
        for token in document_tokens:
            if token in features_dictionary:
                feature_index = features_dictionary[token]
                bow_vector[feature_index] += 1
        return bow_vector

def main():
    with io.open("./training_data/announcements.json", 'r', encoding="utf-8") as file:
        data_sets = json.load(file)["data"]
    
    try:
        training_sets = [("{0}. {1}".format(data_set["headline"], data_set["document"]), data_set["label"]) for data_set in data_sets]
    except KeyError as e:
        print(f"Training data is deformed, make sure to include a {e} key in the set")
        return
    # for training_set in training_sets:
    #     print(f"document: {training_set[0]} | label: {training_set[1]}")
    test_text = """
        Đăng ký học phần
    """
    result = BagOfWord.predict_from_training_set(training_sets=training_sets, test_document=test_text)
    print(result)

if __name__ == "__main__":
    main()

