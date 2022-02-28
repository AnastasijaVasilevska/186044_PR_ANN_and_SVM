import math
import numpy as np
import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.utils import shuffle
from sklearn import metrics
from sklearn import preprocessing
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn import svm

def first_dataset(n, rounding_factor):
    list_of_elements = [['', 'X', 'Y', 'Class']]

    for x in np.arange(0, 1, n):
        for y in np.arange(0, 1, n):
            if y - ((math.sin(6 * x) / 6) + 0.6) < 0:
                klasa = 'negative'
                attributes = ['', round(x, rounding_factor), round(y, rounding_factor), klasa]
                list_of_elements.append(attributes)
            else:
                klasa = 'positive'
                attributes = ['', round(x, rounding_factor), round(y, rounding_factor), klasa]
                list_of_elements.append(attributes)

    data = np.array(list_of_elements)
    return pd.DataFrame(data=data[1:, 1:], index=data[1:, 0], columns=data[0, 1:])

def second_dataset(n, rounding_factor):
    list_of_elements = [['', 'X', 'Y', 'Class']]
    for x in np.arange(0, 1, n):
        for y in np.arange(0, 1, n):
            if (x < 0.25 and y < 0.25) or (x < 0.25 and y < 0.75 and y > 0.5) or (
                    x < 0.5 and x > 0.25 and y > 0.25 and y < 0.5) or (x < 0.5 and x > 0.25 and y > 0.75) or (
                    x < 0.75 and x > 0.5 and y < 0.25) or (x < 0.75 and x > 0.5 and y < 0.75 and y > 0.5) or (
                    x > 0.75 and y < 0.5 and y > 0.25) or (x > 0.75 and y > 0.75):
                klasa = "black"
                attributes = ['', round(x, rounding_factor), round(y, rounding_factor), klasa]
                list_of_elements.append(attributes)
            else:
                klasa = "white"
                attributes = ['', round(x, rounding_factor), round(y, rounding_factor), klasa]
                list_of_elements.append(attributes)
    data = np.array(list_of_elements)
    return pd.DataFrame(data=data[1:, 1:], index=data[1:, 0], columns=data[0, 1:])

def neural_network_classifier1(neurons, training_features, training_klasa, test_features, test_klasa):
    mlp = MLPClassifier(hidden_layer_sizes=(neurons, ))
    mlp.fit(training_features, training_klasa)

    le = preprocessing.LabelEncoder()
    test_klasa_transformed = le.fit_transform(test_klasa)

    pred_klasa_proba = mlp.predict_proba(test_features)
    pred_klasa = [int(proba[1] > 0.5) for proba in pred_klasa_proba]
    print('MLP accuracy: ', metrics.accuracy_score(test_klasa_transformed, pred_klasa))

def neural_network_classifier2(neurons1, neurons2, training_features, training_klasa, test_features, test_klasa):
    mlp = MLPClassifier(hidden_layer_sizes=(neurons1, neurons2))
    mlp.fit(training_features, training_klasa)

    le = preprocessing.LabelEncoder()
    test_klasa_transformed = le.fit_transform(test_klasa)

    pred_klasa_proba = mlp.predict_proba(test_features)
    pred_klasa = [int(proba[1] > 0.5) for proba in pred_klasa_proba]
    print('MLP Accuracy: ', metrics.accuracy_score(test_klasa_transformed, pred_klasa))

def linear_SVM(C, training_features, training_klasa, test_features, test_klasa):
    lin_svm = svm.LinearSVC(C=C)
    lin_svm.fit(training_features, training_klasa)

    pred_klasa = lin_svm.predict(test_features)
    print("Linear SVM Accuracy: ", metrics.accuracy_score(test_klasa, pred_klasa))

def gaussian_SVM(C, gamma, training_features, training_klasa, test_features, test_klasa):
    g_svm = svm.SVC(kernel='rbf', C=C, gamma=gamma)
    g_svm.fit(training_features, training_klasa)

    pred_klasa = g_svm.predict(test_features)
    print('Gaussian SVM Accuracy: ', metrics.accuracy_score(test_klasa, pred_klasa))

def synthetic_dataset_classifier(dataset, type):
    training_data = dataset[:int(len(dataset) * 0.8)]
    test_data = dataset[int(len(dataset) * 0.8):]

    training_features = training_data[['X', 'Y']]
    training_klasa = training_data['Class']

    test_features = test_data[['X', 'Y']]
    test_klasa = test_data['Class']


    if type == 'mlp':
        print("With 5 neurons in hidden layer:")
        neural_network_classifier1(5, training_features, training_klasa, test_features, test_klasa)
        print("With 10 neurons in hidden layer:")
        neural_network_classifier1(10, training_features, training_klasa, test_features, test_klasa)
        print("With 15 neurons in hidden layer:")
        neural_network_classifier1(20, training_features, training_klasa, test_features, test_klasa)
        print("With 2 hidden layers:")
        neural_network_classifier2(5, 2, training_features, training_klasa, test_features, test_klasa)
    if type == 'lsvm':
        print("C = 1.0:")
        linear_SVM(1.0, training_features, training_klasa, test_features, test_klasa)
        print("C = 2.0:")
        linear_SVM(2.0, training_features, training_klasa, test_features, test_klasa)
        print("C = 4.0:")
        linear_SVM(4.0, training_features, training_klasa, test_features, test_klasa)
    if type == 'gsvm':
        print("C = 1.0 and Gamma = scale:")
        gaussian_SVM(1.0, 'scale', training_features, training_klasa, test_features, test_klasa)
        print("C = 1.0 and Gamma = auto:")
        gaussian_SVM(1.0, 'auto', training_features, training_klasa, test_features, test_klasa)
        print("C = 2.0 and Gamma = scale:")
        gaussian_SVM(2.0, 'scale', training_features, training_klasa, test_features, test_klasa)
        print("C = 2.0 and Gamma = auto:")
        gaussian_SVM(2.0, 'auto', training_features, training_klasa, test_features, test_klasa)


def iris_cross_validation(dataset, type):
    features = dataset.iloc[:, :-1]
    klasa = dataset.iloc[:, -1:]
    klasa = np.ravel(klasa)
    cv = KFold(n_splits=10, random_state=1, shuffle=True)

    if type == 'mlp':
        print("With 5 neurons in hidden layer:")
        mlp = MLPClassifier(hidden_layer_sizes=(5,))
        scores_mlp = cross_val_score(mlp, features, klasa, scoring='accuracy', cv=cv, n_jobs=-1)
        print('MLP Accuracy: ', (np.mean(scores_mlp)))
        print("With 10 neurons in hidden layer:")
        mlp = MLPClassifier(hidden_layer_sizes=(10,))
        scores_mlp = cross_val_score(mlp, features, klasa, scoring='accuracy', cv=cv, n_jobs=-1)
        print('MLP Accuracy: ', (np.mean(scores_mlp)))
        print("With 15 neurons in hidden layer:")
        mlp = MLPClassifier(hidden_layer_sizes=(15,))
        scores_mlp = cross_val_score(mlp, features, klasa, scoring='accuracy', cv=cv, n_jobs=-1)
        print('MLP Accuracy: ', (np.mean(scores_mlp)))
        print("With 2 hidden layers:")
        mlp = MLPClassifier(hidden_layer_sizes=(5, 2))
        scores_mlp = cross_val_score(mlp, features, klasa, scoring='accuracy', cv=cv, n_jobs=-1)
        print('MLP Accuracy: ', (np.mean(scores_mlp)))
    if type == 'lsvm':
        print("C = 1.0:")
        lin_svm = svm.LinearSVC(C=1.0)
        scores_linsvm = cross_val_score(lin_svm, features, klasa, scoring='accuracy', cv=cv, n_jobs=-1)
        print('Linear SVM Accuracy: ', (np.mean(scores_linsvm)))
        print("C = 2.0:")
        lin_svm = svm.LinearSVC(C=2.0)
        scores_linsvm = cross_val_score(lin_svm, features, klasa, scoring='accuracy', cv=cv, n_jobs=-1)
        print('Linear SVM Accuracy: ', (np.mean(scores_linsvm)))
        print("C = 4.0:")
        lin_svm = svm.LinearSVC(C=4.0)
        scores_linsvm = cross_val_score(lin_svm, features, klasa, scoring='accuracy', cv=cv, n_jobs=-1)
        print('Linear SVM Accuracy: ', (np.mean(scores_linsvm)))
    if type == 'gsvm':
        print("C = 1.0 and Gamma = scale:")
        g_svm = svm.SVC(kernel='rbf', C=1.0, gamma='scale')
        scores_gsvm = cross_val_score(g_svm, features, klasa, scoring='accuracy', cv=cv, n_jobs=-1)
        print('Gaussian SVM Accuracy: ', (np.mean(scores_gsvm)))
        print("C = 1.0 and Gamma = auto:")
        g_svm = svm.SVC(kernel='rbf', C=1.0, gamma='auto')
        scores_gsvm = cross_val_score(g_svm, features, klasa, scoring='accuracy', cv=cv, n_jobs=-1)
        print('Gaussian SVM Accuracy: ', (np.mean(scores_gsvm)))
        print("C = 2.0 and Gamma = scale:")
        g_svm = svm.SVC(kernel='rbf', C=2.0, gamma='scale')
        scores_gsvm = cross_val_score(g_svm, features, klasa, scoring='accuracy', cv=cv, n_jobs=-1)
        print('Gaussian SVM Accuracy: ', (np.mean(scores_gsvm)))
        print("C = 2.0 and Gamma = auto:")
        g_svm = svm.SVC(kernel='rbf', C=2.0, gamma='auto')
        scores_gsvm = cross_val_score(g_svm, features, klasa, scoring='accuracy', cv=cv, n_jobs=-1)
        print('Gaussian SVM Accuracy: ', (np.mean(scores_gsvm)))

def letter_recognition_classifier(letter_recognition, type):
    training_data = letter_recognition[:int(len(letter_recognition) * 0.66)]
    test_data = letter_recognition[int(len(letter_recognition) * 0.66):]

    training_features = training_data.iloc[:, 1:]
    training_klasa = training_data[0]

    test_features = test_data.iloc[:, 1:]
    test_klasa = test_data[0]

    if type == 'mlp':
        print("With 5 neurons in hidden layer:")
        neural_network_classifier1(5, training_features, training_klasa, test_features, test_klasa)
        print("With 10 neurons in hidden layer:")
        neural_network_classifier1(10, training_features, training_klasa, test_features, test_klasa)
        print("With 15 neurons in hidden layer:")
        neural_network_classifier1(20, training_features, training_klasa, test_features, test_klasa)
        print("With 2 hidden layers:")
        neural_network_classifier2(5, 2, training_features, training_klasa, test_features, test_klasa)
    if type == 'lsvm':
        print("C = 1.0:")
        linear_SVM(1.0, training_features, training_klasa, test_features, test_klasa)
        print("C = 2.0:")
        linear_SVM(2.0, training_features, training_klasa, test_features, test_klasa)
        print("C = 4.0:")
        linear_SVM(4.0, training_features, training_klasa, test_features, test_klasa)
    if type == 'gsvm':
        print("C = 1.0 and Gamma = scale:")
        gaussian_SVM(1.0, 'scale', training_features, training_klasa, test_features, test_klasa)
        print("C = 1.0 and Gamma = auto:")
        gaussian_SVM(1.0, 'auto', training_features, training_klasa, test_features, test_klasa)
        print("C = 2.0 and Gamma = scale:")
        gaussian_SVM(2.0, 'scale', training_features, training_klasa, test_features, test_klasa)
        print("C = 2.0 and Gamma = auto:")
        gaussian_SVM(2.0, 'auto', training_features, training_klasa, test_features, test_klasa)

sin_function1       = shuffle(first_dataset(0.1, 2))
sin_function2       = shuffle(first_dataset(0.01, 4))
sin_function3       = shuffle(first_dataset(0.001, 6))
chess_board1        = shuffle(second_dataset(0.1, 2))
chess_board2        = shuffle(second_dataset(0.01, 4))
chess_board3        = shuffle(second_dataset(0.001, 6))
iris                = pd.read_csv('iris.csv', header=None)
letter_recognition  = pd.read_csv('letter-recognition.csv', header=None)


print("Sine Function with 100 samples")
print("\n")
print("Neural Network Classification:")
synthetic_dataset_classifier(sin_function1, 'mlp')
print("\n")
print("Linear SVM Classification:")
synthetic_dataset_classifier(sin_function1, 'lsvm')
print("\n")
print("Gaussian SVM Classification:")
synthetic_dataset_classifier(sin_function1, 'gsvm')


print("=======================================================")

print("Sine Function with 10000 samples")
print("\n")
print("Neural Network Classification:")
synthetic_dataset_classifier(sin_function2, 'mlp')
print("\n")
print("Linear SVM Classification:")
synthetic_dataset_classifier(sin_function2, 'lsvm')
print("\n")
print("Gaussian SVM Classification:")
synthetic_dataset_classifier(sin_function2, 'gsvm')


print("=======================================================")

print("Sine Function with 1000000 samples")
print("\n")
print("Neural Network Classification:")
synthetic_dataset_classifier(sin_function3, 'mlp')
print("\n")
print("Linear SVM Classification:")
synthetic_dataset_classifier(sin_function3, 'lsvm')
print("\n")
print("Gaussian SVM Classification:")
synthetic_dataset_classifier(sin_function3, 'gsvm') #check this out mnogu vreme mu trebase



print("=======================================================")

print("Chessboard with 100 samples")
print("\n")
print("Neural Network Classification:")
synthetic_dataset_classifier(chess_board1, 'mlp')
print("\n")
print("Linear SVM Classification:")
synthetic_dataset_classifier(chess_board1, 'lsvm')
print("\n")
print("Gaussian SVM Classification:")
synthetic_dataset_classifier(chess_board1, 'gsvm')

print("=======================================================")

print("Chessboard with 10000 samples")
print("\n")
print("Neural Network Classification:")
synthetic_dataset_classifier(chess_board2, 'mlp')
print("\n")
print("Linear SVM Classification:")
synthetic_dataset_classifier(chess_board2, 'lsvm')
print("\n")
print("Gaussian SVM Classification:")
synthetic_dataset_classifier(chess_board2, 'gsvm')


print("=======================================================")

print("Chessboard with 1000000 samples")
print("\n")
print("Neural Network Classification:")
synthetic_dataset_classifier(chess_board3, 'mlp')
print("\n")
print("Linear SVM Classification:")
synthetic_dataset_classifier(chess_board3, 'lsvm')
print("\n")
print("Gaussian SVM Classification:")
synthetic_dataset_classifier(chess_board3, 'gsvm')


print("=======================================================")

print("Iris Dataset 10 Fold Cross-Validation")
print("\n")
print("Neural Network Classification:")
iris_cross_validation(iris, 'mlp')
print("\n")
print("Linear SVM Classification:")
iris_cross_validation(iris, 'lsvm')
print("\n")
print("Gaussian SVM Classification:")
iris_cross_validation(iris, 'gsvm')


print("=======================================================")

print("Letter Recognition Classification")
print("\n")
print("Neural Network Classification:")
letter_recognition_classifier(letter_recognition, 'mlp')
print("\n")
print("Linear SVM Classification:")
letter_recognition_classifier(letter_recognition, 'lsvm')
print("\n")
print("Gaussian SVM Classification:")
letter_recognition_classifier(letter_recognition, 'gsvm')

