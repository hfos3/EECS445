"""EECS 445 - Fall 2023.

Project 1
"""

import itertools
import string
import warnings

import numpy as np
import pandas as pd
from gensim.models import Word2Vec
from helper import *
from matplotlib import pyplot as plt
from sklearn import metrics
from sklearn.exceptions import ConvergenceWarning
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC, LinearSVC

warnings.simplefilter(action="ignore", category=FutureWarning)
warnings.simplefilter(action="ignore", category=ConvergenceWarning)

np.random.seed(445)



def extract_word(input_string):
    """Preprocess review text into list of tokens.

    Convert input string to lowercase, replace punctuation with spaces, and split along whitespace.
    Return the resulting array.

    E.g.
    > extract_word("I love EECS 445. It's my favorite course!")
    > ["i", "love", "eecs", "445", "it", "s", "my", "favorite", "course"]

    Input:
        input_string: text for a single review
    Returns:
        a list of words, extracted and preprocessed according to the directions
        above.
    """
    lower = input_string.lower()
    clean = ''.join([' ' if c in string.punctuation else c for c in lower])
    return (clean.split())
    ##: Implement this function

#['It','s','a','test','sentence','does','it','look','correct']
def extract_dictionary(df):
    """Map words to index.

    Reads a pandas dataframe, and returns a dictionary of distinct words
    mapping from each distinct word to its index (ordered by when it was
    found).

    E.g., with input:
        | reviewText                    | label | ... |
        | It was the best of times.     |  1    | ... |
        | It was the blurst of times.   | -1    | ... |

    The output should be a dictionary of indices ordered by first occurence in
    the entire dataset:
        {
           it: 0,
           was: 1,
           the: 2,
           best: 3,
           of: 4,
           times: 5,
           blurst: 6
        }
    The index should be autoincrementing, starting at 0.

    Input:
        df: dataframe/output of load_data()
    Returns:
        a dictionary mapping words to an index
    """
    i = 0
    word_dict = {}
    for row in df.itertuples():
        arr = extract_word(row.reviewText)
        #print(arr)
        for word in arr:
            if word not in word_dict.keys():
                word_dict[word] = i
                i += 1
    #print(len(word_dict))
    return word_dict


def generate_feature_matrix(df, word_dict):
    """Create matrix of feature vectors for dataset.

    Reads a dataframe and the dictionary of unique words to generate a matrix
    of {1, 0} feature vectors for each review. For each review, extract a token
    list and use word_dict to find the index for each token in the token list.
    If the token is in the dictionary, set the corresponding index in the review's
    feature vector to 1. The resulting feature matrix should be of dimension
    (# of reviews, # of words in dictionary).

    Input:
        df: dataframe that has the text and labels
        word_dict: dictionary of words mapping to indices
    Returns:
        a numpy matrix of dimension (# of reviews, # of words in dictionary)
    """
    number_of_reviews = df.shape[0]
    number_of_words = len(word_dict)
    feature_matrix = np.zeros((number_of_reviews, number_of_words))
    r = 0
    for row in df.reviewText:
        s = extract_word(row)
        for word in s:
            index = word_dict.get(word,-1)
            if index != -1:
                feature_matrix[r][index] = 1
        r += 1
    #print("ANS IS", feature_matrix.sum() / r)
    return feature_matrix




def performance(y_true, y_pred, metric="accuracy"):
    """Calculate performance metrics.

    Performance metrics are evaluated on the true labels y_true versus the
    predicted labels y_pred.

    Input:
        y_true: (n,) array containing known labels
        y_pred: (n,) array containing predicted scores
        metric: string specifying the performance metric (default='accuracy'
                 other options: 'f1-score', 'auroc', 'precision', 'sensitivity',
                 and 'specificity')
    Returns:
        the performance as an np.float64
    """
    tn,fp,fn,tp=0,0,0,0
   # print(y_true)
    #print(y_pred)
    #for i,l in enumerate(y_true):
     #   pred = y_pred[i]
      #  if pred == l:
        #    if pred == -1:
       #         tn += 1
         #   else:
          #      tp += 1
        #else:
         #   if pred == 1:
          #      fp += 1
           # else:
            #    fn += 1
            
    confusion=metrics.confusion_matrix(y_true,y_pred,labels=[1,-1])
    #print("confusion is", confusion)
    tp,fn,fp,tn=confusion[0,0],confusion[0,1],confusion[1,0],confusion[1,1]
    #print(tp,fn,fp,tp)
    if(metric == "accuracy"):
        #return metrics.accuracy_score(y_true,y_pred)
        return (tp+tn)/(tp+fn+fp+tn)
    if(metric == "f1-score"):
        return (2*tp)/((2*tp)+fp+fn)
        #return metrics.f1_score(y_true,y_pred)
    if(metric == "precision"):
        if(tp+fp == 0):
            return 0
        return tp/(tp+fp)
    if(metric == "sensitivity"):
        if(tp+fn == 0):
            return 0
        return tp/(tp+fn)
    if(metric == "specificity"):
        if(tn + fp == 0):
            return 0
        x = tn/(tn+fp)
        return x
    return -5.0#should never happen

    # See the sklearn.metrics documentation for pointers on how to implement
    # the requested metrics.

def auroc_performance(y_true,y_pred):
    return metrics.roc_auc_score(y_true,y_pred)

def cv_performance(clf, X, y, k=5, metric="accuracy"):
    """Split data into k folds and run cross-validation.

    Splits the data X and the labels y into k-folds and runs k-fold
    cross-validation: for each fold i in 1...k, trains a classifier on
    all the data except the ith fold, and tests on the ith fold.
    Calculates and returns the k-fold cross-validation performance metric for
    classifier clf by averaging the performance across folds.
    Input:
        clf: an instance of SVC()
        X: (n,d) array of feature vectors, where n is the number of examples
           and d is the number of features
        y: (n,) array of binary labels {1,-1}
        k: an int specifying the number of folds (default=5)
        metric: string specifying the performance metric (default='accuracy'
             other options: 'f1-score', 'auroc', 'precision', 'sensitivity',
             and 'specificity')
    Returns:
        average 'test' performance across the k folds as np.float64
    """
    # TODO: Implement this function
    # HINT: You may find the StratifiedKFold from sklearn.model_selection
    # to be useful
    skf = StratifiedKFold(n_splits=k)
    skf.get_n_splits(X,y)
    # Put the performance of the model on each fold in the scores array
    scores = []
    for train, test in skf.split(X, y):
        train_data = [X[i] for i in train]#Matrix
        train_true = [y[i] for i in train]#array
        test_data = [X[i] for i in test]
        test_true = [y[i] for i in test]
        clf.fit(train_data,train_true)
        if(metric == 'auroc'):
            scores.append(auroc_performance(test_true,clf.decision_function(test_data)))#adds score to array
        else:
            scores.append(performance(test_true,clf.predict(test_data),metric))#adds score to array
    return np.array(scores).mean()


def select_param_linear(
    X, y, k=5, metric="accuracy", C_range=[], loss="hinge", penalty="l2", dual=True, minPer=[]
):
    """Search for hyperparameters from the given candidates of linear SVM with
    best k-fold CV performance.

    Sweeps different settings for the hyperparameter of a linear-kernel SVM,
    calculating the k-fold CV performance for each setting on X, y.
    Input:
        X: (n,d) array of feature vectors, where n is the number of examples
        and d is the number of features
        y: (n,) array of binary labels {1,-1}
        k: int specifying the number of folds (default=5)
        metric: string specifying the performance metric (default='accuracy',
             other options: 'f1-score', 'auroc', 'precision', 'sensitivity',
             and 'specificity')
        C_range: an array with C values to be searched over
        loss: string specifying the loss function used (default="hinge",
             other option of "squared_hinge")
        penalty: string specifying the penalty type used (default="l2",
             other option of "l1")
        dual: boolean specifying whether to use the dual formulation of the
             linear SVM (set True for penalty "l2" and False for penalty "l1")
    Returns:
        the parameter value for a linear-kernel SVM that maximizes the
        average 5-fold CV performance.
    """
    # TODO: Implement this function
    # HINT: You should be using your cv_performance function here
    # to evaluate the performance of each SVM
    #print(C_range)
    outputs = [(cv_performance(LinearSVC(penalty=penalty,loss=loss,dual=dual,C=c,random_state=445),X,y,k,metric)) for c in C_range]
    target = np.argmax(outputs)
    minPer.append(outputs[target])
    return C_range[target]
    

def plot_weight(X, y, penalty, C_range, loss, dual):
    """Create a plot of the L0 norm learned by a classifier for each C in C_range.

    Input:
        X: (n,d) array of feature vectors, where n is the number of examples
        and d is the number of features
        y: (n,) array of binary labels {1,-1}
        penalty: string for penalty type to be forwarded to the LinearSVC constructor
        C_range: list of C values to train a classifier on
        loss: string for loss function to be forwarded to the LinearSVC constructor
        dual: whether to solve the dual or primal optimization problem, to be
            forwarded to the LinearSVC constructor
    Returns: None
        Saves a plot of the L0 norms to the filesystem.
    """
    norm0 = []
    for c in C_range:
        svm = LinearSVC(penalty=penalty,loss=loss,dual=dual,C=c,random_state=445)
        svm.fit(X,y)
        n = 0
        for arr in svm.coef_:
            for i in arr:
                if i != 0:
                    n += 1
        norm0.append(n)
    # : Implement this part of the function
    # Here, for each value of c in C_range, you should
    # append to norm0 the L0-norm of the theta vector that is learned
    # when fitting an L2- or L1-penalty, degree=1 SVM to the data (X, y)

    plt.plot(C_range, norm0)
    plt.xscale("log")
    plt.legend(["L0-norm"])
    plt.xlabel("Value of C")
    plt.ylabel("Norm of theta")
    plt.title("Norm-" + penalty + "_penalty.png")
    plt.savefig("Norm-" + penalty + "_penalty.png")
    plt.close()


def select_param_quadratic(X, y, k=5, metric="accuracy", param_range=[]):
    """Search for hyperparameters from the given candidates of quadratic SVM
    with best k-fold CV performance.

    Sweeps different settings for the hyperparameters of a quadratic-kernel SVM,
    calculating the k-fold CV performance for each setting on X, y.
    Input:
        X: (n,d) array of feature vectors, where n is the number of examples
           and d is the number of features
        y: (n,) array of binary labels {1,-1}
        k: an int specifying the number of folds (default=5)
        metric: string specifying the performance metric (default='accuracy'
                 other options: 'f1-score', 'auroc', 'precision', 'sensitivity',
                 and 'specificity')
        param_range: a (num_param, 2)-sized array containing the
            parameter values to search over. The first column should
            represent the values for C, and the second column should
            represent the values for r. Each row of this array thus
            represents a pair of parameters to be tried together.
    Returns:
        The parameter values for a quadratic-kernel SVM that maximize
        the average 5-fold CV performance as a pair (C,r)
    """
    #: Implement this function
    # Hint: This will be very similar to select_param_linear, except
    # the type of SVM model you are using will be different...
    #Grid Search
    best_C_val, best_r_val = 0.0, 0.0
    target = 0
    for c in param_range:
        for r in param_range:
            svm = SVC(kernel='poly',degree = 2, C = c[0], coef0 = r[1], gamma = 'auto')
            #svm.fit(X,y)
            n = cv_performance(svm, X, y, metric='auroc')
            if n > target:
                best_C_val,best_r_val,target = c[0],r[1],n
    return best_C_val, best_r_val,target
def randSearch(X,y,metric='auroc'):
    #ii:
    #Random Search
    rng = np.random.default_rng()
    param_range = []
    for _ in range(25):
        c_exp = 3 * (rng.random()-.5)
        r_exp = 3 * (rng.random()-.5)
        param_range.append([10**c_exp,10**r_exp])
    randC,randR,randn = select_param_quadratic(X,y,metric=metric,param_range=param_range)
    return randC,randR,randn

def train_word2vec(fname):
    """
    Train a Word2Vec model using the Gensim library.
    First, iterate through all reviews in the dataframe, run your extract_word() function on each review, and append the result to the sentences list.
    Next, instantiate an instance of the Word2Vec class, using your sentences list as a parameter.
    Return the Word2Vec model you created.
    """
    df = load_data(fname)
    sentences = []
    for r in df['reviewText']:
        s = extract_word(r)
        if len(s) > 0:
            sentences.append(s)
    model = Word2Vec(sentences=sentences,workers=1)
    return model


def compute_association(fname, w, A, B):
    """
    Inputs:
        - fname: name of the dataset csv
        - w: a word represented as a string
        - A and B: sets that each contain one or more English words represented as strings
    Output: Return the association between w, A, and B as defined in the spec
    """
    model = train_word2vec(fname)
    # First, we need to find a numerical representation for the English language words in A and B
    # : Complete words_to_array(), which returns a 2D Numpy Array where the ith row is the embedding vector for the ith word in the input set.
    def words_to_array(set):
        embeddings = []
        for word in set:
            embedding = model.wv[word]
        embeddings.append(embedding)
        return embeddings
    #   Complete cosine_similarity(), which returns a 1D Numpy Array where the ith element is the cosine similarity
    #      between the word embedding for w and the ith embedding in the array representation of the input set
    def cosine_similarity(set):
        array = words_to_array(set)
        similar = metrics.pairwise.cosine_similarity(array,[model.wv[w]])
        return similar.mean()
    return cosine_similarity(A) - cosine_similarity(B)
        

    # TODO: Return the association between w, A, and B.
    #      Compute this by finding the difference between the mean cosine similarity between w and the words in A, and the mean cosine similarity between w and the words in B
    return None



def main():
    # Read binary data
    # NOTE: THE VALUE OF dictionary_binary WILL NOT BE CORRECT UNTIL YOU HAVE IMPLEMENTED
    #       extract_dictionary, AND THE VALUES OF X_train, Y_train, X_test, AND Y_test
    #       WILL NOT BE CORRECT UNTIL YOU HAVE IMPLEMENTED extract_dictionary AND
    #       generate_feature_matrix
    fname = "data/dataset.csv"
    X_train, Y_train, X_test, Y_test, dictionary_binary = get_split_binary_data(
        fname="data/dataset.csv"
    )
    IMB_features, IMB_labels, IMB_test_features, IMB_test_labels = get_imbalanced_data(
        dictionary_binary, fname="data/dataset.csv"
    )
    word_counts = X_train.sum(axis = 0)
    most = np.argmax(word_counts)
    for key,value in dictionary_binary.items():
        if value == most:
            print("Most Common Word: ",key)
            break
    print("UNIQUE WORDS: ", len(dictionary_binary))
    print("AVERAGE IS", word_counts.sum() / X_train.shape[0])
    #q3
    C_range = np.array([.001,.01,.1,1,10,100,1000])
    bestC = []
    metrics = ['accuracy','f1-score','auroc', 'precision', 'sensitivity', 'specificity']
    for m in metrics:
        minPer=[]
        report = select_param_linear(X=X_train,y=Y_train,metric=m,C_range=C_range,minPer=minPer)
        print(report)
        print(m)
        print(report)
        print(minPer[0])
        print()
        bestC.append(report)
    for i in range(len(metrics)):
        print("Metric: ",metrics[i])
        print("c: ",bestC[i])
        clf = LinearSVC(loss="hinge", penalty="l2", dual=True,C=bestC[i],random_state=445)
        clf.fit(X_train,Y_train)
        if metrics[i] == 'auroc':
            print("score: ",auroc_performance(Y_test,clf.decision_function(X_test)))
        else:
            print("score: ",performance(Y_test,clf.predict(X_test),metrics[i]))
        print()
    #31d
    modC = [c for c in C_range if c <= 1]
    plot_weight(X_train,Y_train,C_range=modC,loss='hinge',dual=True,penalty='l2')
#FIGURE OUT BAR PLOT SOMEHOW
    svm = LinearSVC(loss='hinge',penalty='l2',dual=True,C=.1,random_state=445)
    svm.fit(X_train,Y_train)
    sorted = svm.coef_.argsort()
    most = sorted[-5:]
    least = sorted[:5]
    
    print("MOST COMMON:",most)
    print("LEAST COMMON: ",least)
    #31e
#    svm = LinearSVC(penalty='l2',loss='hinge',dual=True,C=.1,random_state=445)
 #   svm.fit(X_train,Y_train)
  #  plt.bar(C_range, norm0)
   # plt.xscale("log")
    #plt.legend(["L0-norm"])
    #plt.xlabel("Value of C")
    #plt.ylabel("Norm of theta")
    #plt.title("Norm-" + penalty + "_penalty.png")
    #plt.savefig("Norm-" + penalty + "_penalty.png")
    #plt.close()
    
    #3.2.a(recent)
    minPer = []
    selection = select_param_linear(X_train,Y_train,metric='auroc',C_range=modC,loss='squared_hinge',penalty='l1',dual=False,minPer=minPer)
    print("3.2A: ",selection)
    print(minPer)
    print(auroc_performance(Y_test,LinearSVC(penalty='l1',dual=False).fit(X_train,Y_train).decision_function(X_test)))
    #3.2.b(recent)
    plot_weight(X_train,Y_train,penalty='l1',C_range=modC,loss='squared_hinge',dual=False)

    #3.3a
    param_range = [
        [.01,.01],
        [.1,.1],
        [1,1],
        [10,10],
        [100,100],
        [1000,1000],
    ]
    #print(param_range)
    #print(param_range)
    #bestC, bestR,n = select_param_quadratic(X=X_train,y=Y_train,metric='auroc',param_range=param_range)
    #print("grid search: ",bestC,bestR,n)
    #randC,randR,randn = randSearch(X_train,Y_train)
    #print("rand search: ", randC,randR,randn)
    #4.1c
    print("IMBALANCED WEIGHTS")
    svm = LinearSVC(C=.01,class_weight={-1:1,1:10})
    svm.fit(X=X_train,y=Y_train)
    for m in metrics:
        y_pred = svm.predict(X_test)
        n = 0
        if m == 'auroc':
            y_pred = svm.decision_function(X_test)
            report = auroc_performance(Y_test,y_pred)
        else:
            report = performance(y_true=Y_test,y_pred=y_pred,metric=m)
        print(m)
        print(report)
        print()

    #4.2a
    svm = LinearSVC(C=.01,loss='hinge',class_weight={-1:1,1:1})
    svm.fit(IMB_features,IMB_labels)
    clf = LinearSVC(C=.01)
    clf.fit(IMB_features,IMB_labels)
    #print(IMB_test_features)
    print(IMB_test_labels)
    pred = clf.predict(IMB_test_features)
    for m in metrics:
        if m == 'auroc':
            x = auroc_performance(IMB_test_labels,svm.decision_function(IMB_test_features))
            print(m," : ",x)
        else:
            x = performance(IMB_test_labels,clf.predict(IMB_test_features),m)
            print(m," : ",x)
    
    # Read multiclass data
    # : Question 6: Apply a classifier to heldout features, and then use
    #       generate_challenge_labels to print the predicted labels
    #5.1
    Nactor,Nactress = count_actors_and_actresses(fname=fname)
    print("actor",Nactor)
    print("actress",Nactress)
    #plot_actors_and_actresses(fname=fname,x_label='label')
    plot_actors_and_actresses(fname=fname,x_label='rating')
    #5.2
    embed = train_word2vec(fname="data/dataset.csv")
    print("EMBEDDING:",embed.wv['actor'])
    print("SIMILAR TO PLOT",embed.wv.most_similar('plot',topn=5))
    #5.3
    x = compute_association(fname,'talented',['her','woman','women'],['him','man','men'])
    print("ASSOCIATION:",x)
    (
        multiclass_features,
        multiclass_labels,
        multiclass_dictionary,
    ) = get_multiclass_training_data()
    clf = LinearSVC(loss="hinge", penalty="l2", dual=True,C=.1,multi_class='ovr',class_weight={-1:2,0:1,1:2})
    clf.fit(multiclass_features,multiclass_labels)
    heldout_features = get_heldout_reviews(multiclass_dictionary)
    prediction = clf.predict(heldout_features)
    generate_challenge_labels(prediction,'hafoster')
    return 0
if __name__ == "__main__":
    main()
