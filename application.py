import xlrd
import xlsxwriter
from tkinter import *
from tkinter import ttk
'''
from tkinter import *
from tkinter import ttk
import time
from datetime import date
import xlrd
import tensorflow as tf, sys
import cv2
import time
from tkinter import filedialog
import tkinter.messagebox
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from math import log, sqrt
import pandas as pd
import numpy as np
import re
##joblib.dump(trainData,'TRAIN.joblib.pkl')
##joblib.dump(testData,'TEST.joblib.pkl')
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from apiclient.discovery import build
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_moons, make_circles, make_classification
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
'''

# arguments to be passed to build function
DEVELOPER_KEY = "AIzaSyCReNA7gPbFdimdisbbpH-5Rs9PNluEYig"
YOUTUBE_API_SERVICE_NAME = "youtube"
YOUTUBE_API_VERSION = "v3"


def Preprocess_Comments(Comments_Cont, lower_case=True, stem=True, stop_words=True, gram=2):
    if lower_case:
        Comments_Cont = Comments_Cont.lower()
    words = word_tokenize(Comments_Cont)
    words = [w for w in words if len(w) > 2]
    if gram > 1:
        w = []
        for i in range(len(words) - gram + 1):
            w += [' '.join(words[i:i + gram])]
        return w
    if stop_words:
        sw = stopwords.words('english')
        words = [word for word in words if word not in sw]
    if stem:
        stemmer = PorterStemmer()
        words = [stemmer.stem(word) for word in words]
    return words


def Comparisisioon():
    h = .02  # step size in the mesh
    names = ["Nearest Neighbors", "Linear SVM", "Naive Bayes"]
    classifiers = [KNeighborsClassifier(3), SVC(kernel="linear", C=0.025), GaussianNB()]

    X, y = make_classification(n_features=2, n_redundant=0, n_informative=2, random_state=1, n_clusters_per_class=1)
    rng = np.random.RandomState(2)
    X += 2 * rng.uniform(size=X.shape)
    linearly_separable = (X, y)
    datasets = [make_moons(noise=0.3, random_state=0)]

    figure = plt.figure(figsize=(10, 3))
    i = 1
    # iterate over datasets
    for ds_cnt, ds in enumerate(datasets):
        # preprocess dataset, split into training and test part
        X, y = ds
        print('TRAIN X', np.shape(X))
        print('TRAIN Y', np.shape(y))
        X = StandardScaler().fit_transform(X)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=42)

        x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
        y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                             np.arange(y_min, y_max, h))

        # just plot the dataset first
        cm = plt.cm.RdBu
        cm_bright = ListedColormap(['#FF0000', '#0000FF'])
        ax = plt.subplot(len(datasets), len(classifiers) + 1, i)
        if ds_cnt == 0:
            ax.set_title("Input data")
        # Plot the training points
        ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=cm_bright,
                   edgecolors='k')
        # Plot the testing points
        ax.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=cm_bright, alpha=0.6,
                   edgecolors='k')
        ax.set_xlim(xx.min(), xx.max())
        ax.set_ylim(yy.min(), yy.max())
        ax.set_xticks(())
        ax.set_yticks(())
        i += 1

        # iterate over classifiers
        joblib.dump(X_train, 'X_train.joblib.pkl')
        joblib.dump(y_train, 'y_train.joblib.pkl')
        joblib.dump(X_test, 'X_test.joblib.pkl')
        joblib.dump(y_test, 'y_test.joblib.pkl')
        for name, clf in zip(names, classifiers):
            ax = plt.subplot(len(datasets), len(classifiers) + 1, i)
            clf.fit(X_train, y_train)
            score = clf.score(X_test, y_test)

            # Plot the decision boundary. For that, we will assign a color to each
            # point in the mesh [x_min, x_max]x[y_min, y_max].
            if hasattr(clf, "decision_function"):
                Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
            else:
                Z = clf.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]

            # Put the result into a color plot
            Z = Z.reshape(xx.shape)
            ax.contourf(xx, yy, Z, cmap=cm, alpha=.8)

            # Plot the training points
            ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=cm_bright,
                       edgecolors='k')
            # Plot the testing points
            ax.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=cm_bright,
                       edgecolors='k', alpha=0.6)

            ax.set_xlim(xx.min(), xx.max())
            ax.set_ylim(yy.min(), yy.max())
            ax.set_xticks(())
            ax.set_yticks(())
            if ds_cnt == 0:
                ax.set_title(name)
            ax.text(xx.max() - .3, yy.min() + .3, ('%.2f' % score).lstrip('0'),
                    size=15, horizontalalignment='right')
            i += 1

    ##plt.tight_layout()
    ##plt.show()


class Classify_Cascade(object):
    def __init__(sklearn, trainData, method='Proposed'):
        sklearn.Comments, sklearn.labels = trainData['Comments_Cont'], trainData['label']
        sklearn.method = method

    def train(sklearn):
        sklearn.calc_PROP_FV()
        if sklearn.method == 'Proposed':
            sklearn.calc_TF_IDF()

    def calc_PROP_FV(sklearn):
        noOfComments_Conts = sklearn.Comments.shape[0]
        sklearn.spam_Comments, sklearn.ham_Comments = sklearn.labels.value_counts()[1], sklearn.labels.value_counts()[0]
        sklearn.total_Comments = sklearn.spam_Comments + sklearn.ham_Comments
        sklearn.spam_words = 0
        sklearn.ham_words = 0
        sklearn.tf_spam = dict()
        sklearn.tf_ham = dict()
        sklearn.idf_spam = dict()
        sklearn.idf_ham = dict()
        for i in range(noOfComments_Conts):
            Comments_Cont_processed = Preprocess_Comments(sklearn.Comments[i])
            count = list()
            for word in Comments_Cont_processed:
                if sklearn.labels[i]:
                    sklearn.tf_spam[word] = sklearn.tf_spam.get(word, 0) + 1
                    sklearn.spam_words += 1
                else:
                    sklearn.tf_ham[word] = sklearn.tf_ham.get(word, 0) + 1
                    sklearn.ham_words += 1
                if word not in count:
                    count += [word]
            for word in count:
                if sklearn.labels[i]:
                    sklearn.idf_spam[word] = sklearn.idf_spam.get(word, 0) + 1
                else:
                    sklearn.idf_ham[word] = sklearn.idf_ham.get(word, 0) + 1

    def calc_TF_IDF(sklearn):
        sklearn.Fvspam = dict()
        sklearn.prob_ham = dict()
        sklearn.sum_tf_idf_spam = 0
        sklearn.sum_tf_idf_ham = 0
        for word in sklearn.tf_spam:
            sklearn.Fvspam[word] = (sklearn.tf_spam[word]) * log((sklearn.spam_Comments + sklearn.ham_Comments) \
                                                                 / (sklearn.idf_spam[word] + sklearn.idf_ham.get(word,
                                                                                                                 0)))
            sklearn.sum_tf_idf_spam += sklearn.Fvspam[word]
        for word in sklearn.tf_spam:
            sklearn.Fvspam[word] = (sklearn.Fvspam[word] + 1) / (
                        sklearn.sum_tf_idf_spam + len(list(sklearn.Fvspam.keys())))

        for word in sklearn.tf_ham:
            sklearn.prob_ham[word] = (sklearn.tf_ham[word]) * log((sklearn.spam_Comments + sklearn.ham_Comments) \
                                                                  / (sklearn.idf_spam.get(word, 0) + sklearn.idf_ham[
                word]))
            sklearn.sum_tf_idf_ham += sklearn.prob_ham[word]
        for word in sklearn.tf_ham:
            sklearn.prob_ham[word] = (sklearn.prob_ham[word] + 1) / (
                        sklearn.sum_tf_idf_ham + len(list(sklearn.prob_ham.keys())))

        sklearn.Fvspam_YTB, sklearn.prob_ham_YTB = sklearn.spam_Comments / sklearn.total_Comments, sklearn.ham_Comments / sklearn.total_Comments

    def classify(sklearn, processed_Comments_Cont):

        YTB_spam, YTB_ham = 0, 0
        for word in processed_Comments_Cont:
            if word in sklearn.Fvspam:
                YTB_spam += log(sklearn.Fvspam[word])
            else:
                if sklearn.method == 'Proposed':
                    YTB_spam -= log(sklearn.sum_tf_idf_spam + len(list(sklearn.Fvspam.keys())))
                else:
                    YTB_spam -= log(sklearn.spam_words + len(list(sklearn.Fvspam.keys())))
            if word in sklearn.prob_ham:
                YTB_ham += log(sklearn.prob_ham[word])
            else:
                if sklearn.method == 'Proposed':
                    YTB_ham -= log(sklearn.sum_tf_idf_ham + len(list(sklearn.prob_ham.keys())))
                else:
                    YTB_ham -= log(sklearn.ham_words + len(list(sklearn.prob_ham.keys())))
            YTB_spam += log(sklearn.Fvspam_YTB)
            YTB_ham += log(sklearn.prob_ham_YTB)
        return YTB_spam >= YTB_ham

    def predict(sklearn, testData):
        result = dict()
        for (i, Comments_Cont) in enumerate(testData):
            processed_Comments_Cont = Preprocess_Comments(Comments_Cont)
            result[i] = int(sklearn.classify(processed_Comments_Cont))
        return result


workbook = xlsxwriter.Workbook('demo.xlsx')
worksheet = workbook.add_worksheet()

# worksheet.set_column('A:A', 20)
bold = workbook.add_format({'bold': True})
worksheet.write('A1', 'USERNAME')
worksheet.write('B1', 'PASSWORD')
worksheet.write('C1', 'MOBILE NUMBER')
worksheet.write('D1', 'ROLL NUMBER')
worksheet.write('E1', 'EMAIL ID')

window = Tk()
window.title("Welcome to YOUTUBE SPAM DETECTION system")
window.geometry('800x500')

tab_control = ttk.Notebook(window)
tab1 = ttk.Frame(tab_control)
tab2 = ttk.Frame(tab_control)
tab3 = ttk.Frame(tab_control)
tab_control.add(tab1, text='STUDENT REGISTRATION')
tab_control.add(tab2, text='TOUTUBE SPAM')


#############################################################################################################################################################
# HEADING
def show_entry_fields():
    print("First Name: %s\nLast Name: %s" % (e1.get(), e2.get()))
    Un = e1.get()
    Pw = e2.get()
    print((Un))
    res = "PERSON " + Un + " IS ADDED"
    lbl1.configure(text=res)
    worksheet.write(str('A' + str(2)), str(Un))
    worksheet.write(str('B' + str(2)), str(Pw))
    workbook.close()


def TST_Face():
    Un = ee1.get()
    Pw = ee2.get()
    VI = ee3.get()
    video_id = VI
    print('USERNAME', Un)
    wb = xlrd.open_workbook('demo.xlsx')
    sheet = wb.sheet_by_index(0)
    Un1 = sheet.cell_value(1, 0)
    Pw1 = sheet.cell_value(1, 1)
    print('UN', Un1);
    print('PW', Pw1);
    if Un == Un1 and Pw == Pw1:
        messagebox.showinfo('LOGIN SUCCESSFUL', 'WELCOME')
    else:
        messagebox.showerror('LOGIN DENIED', 'Wrong Username Or Password')
        window.quit()
        window.destroy()
    # IF LOGIN SUCCESFUL
    lbl21.configure(text="welcome")
    image_path = filedialog.askopenfilename(filetypes=(("BROWSE TRAINING FILE", "*.csv"), ("All files", "*")))
    Comments = pd.read_csv(image_path, encoding='latin-1')
    Comments.drop(['COMMENT_ID', 'AUTHOR', 'DATE'], axis=1, inplace=True)
    Comments.rename(columns={'CLASS': 'labels', 'CONTENT': 'Comments_Cont'}, inplace=True)
    Comments['labels'].value_counts()
    Comments['label'] = Comments['labels'].map({0: 0, 1: 1})
    Comments.drop(['labels'], axis=1, inplace=True)
    totalComments = 300
    trainIndex, testIndex = list(), list()
    for i in range(Comments.shape[0]):
        testIndex += [i]
        trainIndex += [i]
    trainData = Comments.loc[trainIndex]
    testData = Comments.loc[testIndex]
    trainData.reset_index(inplace=True)
    trainData.drop(['index'], axis=1, inplace=True)
    trainData.head()
    testData.reset_index(inplace=True)
    testData.drop(['index'], axis=1, inplace=True)
    testData.head()
    trainData['label'].value_counts()
    testData['label'].value_counts()
    trainData.head()
    trainData['label'].value_counts()
    testData.head()
    testData['label'].value_counts()
    CLF_fused = Classify_Cascade(trainData, 'Proposed')
    CLF_fused.train()
    preds_tf_idf = CLF_fused.predict(testData['Comments_Cont'])
    # Call the comments.list method to retrieve video comments
    youtube = build(YOUTUBE_API_SERVICE_NAME, YOUTUBE_API_VERSION, developerKey=DEVELOPER_KEY)
    max_results = 10
    results = youtube.commentThreads().list(videoId=video_id, part="id,snippet", order="relevance",
                                            textFormat="plainText", maxResults=max_results % 101).execute()
    comments = []
    # Extracting required info from each result
    for result in results['items']:
        comment = {}
        comments.append(comment)
        u = result['snippet']['topLevelComment']['snippet']['textOriginal']
        u = u.encode('unicode-escape').decode('utf-8')
        print(u)
        RESULT = CLF_fused.classify(u)
        print(RESULT);
        if RESULT == 0:
            RESULT = 'NA'
        elif RESULT == 1:
            RESULT = 'SPAM'
        # lbl20.configure(text=u)
        # lbl21.configure(text=RESULT)
        time.sleep(2)


#######################################################################################################
lbl = Label(tab1, text="STUDENT", font=("Arial Bold", 30), foreground=("red"), background=("white"))
lbl.grid(column=0, row=0)
lbl = Label(tab1, text="REGISTRATION", font=("Arial Bold", 30), foreground=("red"), background=("white"))
lbl.grid(column=1, row=0)
lbl = Label(tab1, text="DETAILS", font=("Arial Bold", 30), foreground=("red"), background=("white"))
lbl.grid(column=2, row=0)
# USERNAME & PASSWORD ENTRY BOX
Label(tab1, text="USERNAME", font=("Arial Bold", 15), foreground=("green")).grid(row=1, column=0)
Label(tab1, text="PASSWORD", font=("Arial Bold", 15), foreground=("green")).grid(row=2, column=0)
Label(tab1, text="MOBILE NUMBER", font=("Arial Bold", 15), foreground=("green")).grid(row=3, column=0)
Label(tab1, text="ROLL NUMBER", font=("Arial Bold", 15), foreground=("green")).grid(row=4, column=0)
Label(tab1, text="EMAIL ID", font=("Arial Bold", 15), foreground=("green")).grid(row=5, column=0)
e1 = Entry(tab1)
e2 = Entry(tab1)
e3 = Entry(tab1)
e4 = Entry(tab1)
e5 = Entry(tab1)
e1.grid(row=1, column=1)
e2.grid(row=2, column=1)
e3.grid(row=3, column=1)
e4.grid(row=4, column=1)
e5.grid(row=5, column=1)
lbl1 = Label(tab1, text="  STATUS   ", font=("Arial Bold", 10), foreground=("red"), background=("white"))
lbl1.grid(column=1, row=7)
Button(tab1, text='CANCEL', command=tab1.quit).grid(row=6, column=1, sticky=W, pady=4)
Button(tab1, text='REGISTER', command=show_entry_fields).grid(row=6, column=2, sticky=W, pady=4)
#############################################################################################################################################################
lbl = Label(tab2, text="SPAM", font=("Arial Bold", 30), foreground=("red"), background=("white"))
lbl.grid(column=0, row=0)
lbl = Label(tab2, text="CLASSIFICATION", font=("Arial Bold", 30), foreground=("red"), background=("white"))
lbl.grid(column=1, row=0)
lbl = Label(tab2, text="SYSTEM", font=("Arial Bold", 30), foreground=("red"), background=("white"))
lbl.grid(column=2, row=0)
# USERNAME & PASSWORD ENTRY BOX
Label(tab2, text="USERNAME", font=("Arial Bold", 15), foreground=("green")).grid(row=1, column=0)
Label(tab2, text="PASSWORD", font=("Arial Bold", 15), foreground=("green")).grid(row=2, column=0)
Label(tab2, text="VIDEO ID", font=("Arial Bold", 15), foreground=("green")).grid(row=3, column=0)
ee1 = Entry(tab2)
ee2 = Entry(tab2)
ee3 = Entry(tab2)
ee1.grid(row=1, column=1)
ee2.grid(row=2, column=1)
ee3.grid(row=3, column=1)
lbl21 = Label(tab2, text="  STATUS   ", font=("Arial Bold", 10), foreground=("red"), background=("white"))
lbl21.grid(column=1, row=7)
lbl20 = Label(tab2, text="  COMMENT   ", font=("Arial Bold", 10), foreground=("red"), background=("white"))
lbl20.grid(column=0, row=7)
Button(tab2, text='CANCEL', command=tab2.quit).grid(row=6, column=1, sticky=W, pady=4)
Button(tab2, text='LOGIN', command=TST_Face).grid(row=6, column=2, sticky=W, pady=4)
#############################################################################################################################################################
tab_control.pack(expand=1, fill='both')
window.mainloop()
