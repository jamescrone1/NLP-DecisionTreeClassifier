# Load libraries
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.tree import DecisionTreeClassifier  # Import Decision Tree Classifier
from sklearn.model_selection import train_test_split  # Import train_test_split function
from sklearn import metrics  # Import scikit-learn metrics module for accuracy calculation
from sklearn import preprocessing
import pickle

from code.write_csv_with_features import get_sentences_from_text, \
    sentence_processor_single_cv, get_sentences_from_text_file

file_path = 'C:\\Users\\james\\Documents\\Queens\\Final Year Project\\Dataset\\generatedFiles\\Actual\\'
file_path_model = 'C:\\Users\\james\\Documents\\Queens\\Final Year Project\\Project\\FinalYearProject\\webapp' \
                  '\\decision_tree\\'

col_names = ['sentence no.', 'word', 'is_entity', 'start_index', 'end_index', 'entity_start', 'entity_end',
             'in_dictionary', 'Is University', 'Word Frequency', 'target']
feature_cols_full = ['is_entity', 'entity_start', 'entity_end', 'in_dictionary', 'Word Frequency', 'Is University']
feature_cols_single = ['is_entity']
feature_cols_without_entities = ['is_entity', 'in_dictionary', 'Word Frequency', 'Is University']


def test_accuracy_with_features():
    cv_nlp_csv = pd.read_csv(file_path + "cv_sentence_list_2.csv", names=col_names, nrows=120000, encoding="utf-8")
    new_csv = cv_nlp_csv.drop(['sentence no.', 'word', 'start_index', 'end_index'], axis=1)
    X = new_csv.select_dtypes(exclude=[int])
    le = preprocessing.LabelEncoder()
    X_encoded = encode_categorical_values(X, le)
    build_decision_tree_classifier(X_encoded, feature_cols_full, le)


def test_accuracy_without_features():
    cv_nlp_csv = pd.read_csv(file_path + "cv_sentence_list_2.csv", names=col_names, nrows=120000, encoding="utf-8")
    new_csv = cv_nlp_csv.drop(['sentence no.', 'word', 'start_index', 'end_index', 'entity_start', 'entity_end',
                               'in_dictionary', 'Is University', 'Word Frequency'], axis=1)
    X = new_csv.select_dtypes(exclude=[int])
    le = preprocessing.LabelEncoder()
    X_encoded = encode_categorical_values(X, le)
    build_decision_tree_classifier(X_encoded, feature_cols_single, le)


def predict_values(text):
    loaded_model = pickle.load(open(file_path_model + 'finalized_model.sav', 'rb'))
    sentences = get_sentences_from_text_file()
    sentence_processor_single_cv(sentences)
    cv_nlp_csv = pd.read_csv(file_path + "cv_sentence_list.csv", names=col_names, encoding="utf-8")
    new_csv = cv_nlp_csv.drop(['sentence no.', 'word', 'start_index', 'end_index', 'entity_start', 'entity_end',
                               'target'], axis=1)
    X = new_csv.select_dtypes(exclude=[int])
    le = preprocessing.LabelEncoder()
    X_encoded = encode_categorical_values(X, le)
    X_2 = X_encoded[feature_cols_without_entities]
    cv_pred = loaded_model.predict(X_2)
    return get_inverse_labels(le, cv_pred)


def get_model_accuracy_values(y_test, y_pred):
    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))


def get_inverse_labels(le, y):
    array = le.inverse_transform(y)
    return array


def encode_categorical_values(csv, le):
    X_2 = csv.apply(le.fit_transform)
    return X_2


def build_decision_tree_classifier(table, feature_cols, le):
    X = table[feature_cols]
    y = table.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)
    clf = DecisionTreeClassifier()
    # Train Decision Tree Classifer
    clf = clf.fit(X_train, y_train)
    filename = 'finalized_model.sav'
    pickle.dump(clf, open(filename, 'wb'))
    test_accuracy_of_model(clf, y_test, X_test, le)


def test_accuracy_of_model(clf, y_test, X_test, le):
    y_pred = clf.predict(X_test)
    get_inverse_labels(le, y_pred)
    print("Accuracy:", metrics.accuracy_score(y_test, y_pred))

#
# print("Model without features: ")
# test_accuracy_without_features()
# print("Model with features: ")
# test_accuracy_with_features()
#
# predict_values(file_path)
