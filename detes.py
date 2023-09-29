import streamlit as st
import pandas as pd
import pyttsx3
import re
from sklearn import preprocessing
from sklearn.tree import DecisionTreeClassifier, _tree
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC
import csv
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
training = pd.read_csv('Data/Training.csv')
testing = pd.read_csv('Data/Testing.csv')
cols = training.columns
cols = cols[:-1]
x = training[cols]
y = training['prognosis']
y1 = y
reduced_data = training.groupby(training['prognosis']).max()
le = preprocessing.LabelEncoder()
le.fit(y)
y = le.transform(y)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)
testx = testing[cols]
testy = testing['prognosis']
testy = le.transform(testy)
clf1 = DecisionTreeClassifier()
clf = clf1.fit(x_train, y_train)
clf.feature_names = cols
scores = cross_val_score(clf, x_test, y_test, cv=3)
st.write(scores.mean())
model = SVC()
model.fit(x_train, y_train)
st.write("for svm: ")
st.write(model.score(x_test, y_test))
importances = clf.feature_importances_
indices = np.argsort(importances)[::-1]
features = cols

def readn(nstr):
    engine = pyttsx3.init()
    engine.setProperty('voice', "english+f5")
    engine.setProperty('rate', 130)
    engine.say(nstr)
    engine.runAndWait()
    engine.stop()

severityDictionary = dict()
description_list = dict()
precautionDictionary = dict()

symptoms_dict = {}

for index, symptom in enumerate(x):
    symptoms_dict[symptom] = index

def calc_condition(exp, days):
    sum = 0
    for item in exp:
        if item in severityDictionary:
            sum = sum + severityDictionary[item]
    if ((sum * days) / (len(exp) + 1) > 13):
        st.write("You should seek consultation from a doctor.")
    else:
        st.write("It might not be that bad, but you should take precautions.")

def getDescription():
    global description_list
    with open('MasterData/symptom_Description.csv') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        for row in csv_reader:
            if len(row) == 2:
                _description = {row[0]: row[1]}
                description_list.update(_description)

def getSeverityDict():
    global severityDictionary
    with open('MasterData/symptom_severity.csv') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        try:
            for row in csv_reader:
                if len(row) == 2:
                    symptom, severity = row
                    try:
                        severity = int(severity)
                        severityDictionary[symptom] = severity
                    except ValueError:
                        st.write(f"Invalid severity value for symptom '{symptom}': {severity}. Skipping.")
                else:
                    st.write(f"Skipping row with incorrect format: {row}")
        except:
            pass

def getprecautionDict():
    global precautionDictionary
    with open('MasterData/symptom_precaution.csv') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        for row in csv_reader:
            if len(row) > 1:
                _prec = {row[0]: [row[1], row[2], row[3], row[4]]}
                precautionDictionary.update(_prec)

def getInfo():
    st.write("-----------------------------------HealthCare ChatBot-----------------------------------")
    name = st.text_input("Your Name?")
    st.write("Hello, ", name)

def check_pattern(dis_list, inp):
    pred_list = []
    inp = inp.replace(' ', '_')
    patt = f"{inp}"
    regexp = re.compile(patt)
    pred_list = [item for item in dis_list if regexp.search(item)]
    if len(pred_list) > 0:
        return 1, pred_list
    else:
        return 0, []

def sec_predict(symptoms_exp):
    df = pd.read_csv('Data/Training.csv')
    X = df.iloc[:, :-1]
    y = df['prognosis']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=20)
    rf_clf = DecisionTreeClassifier()
    rf_clf.fit(X_train, y_train)

    symptoms_dict = {symptom: index for index, symptom in enumerate(X)}
    input_vector = np.zeros(len(symptoms_dict))
    for item in symptoms_exp:
        input_vector[[symptoms_dict[item]]] = 1

    return rf_clf.predict([input_vector])

def print_disease(node):
    node = node[0]
    val = node.nonzero()
    disease = le.inverse_transform(val[0])
    return list(map(lambda x: x.strip(), list(disease)))

def tree_to_code(tree, feature_names):
    tree_ = tree.tree_
    feature_name = [
        feature_names[i] if i != _tree.TREE_UNDEFINED else "undefined!"
        for i in tree_.feature
    ]

    chk_dis = ",".join(feature_names).split(",")
    symptoms_present = []

    while True:
        disease_input = st.text_input("Enter the symptom you are experiencing")
        conf, cnf_dis = check_pattern(chk_dis, disease_input)
        if conf == 1:
            st.write("Searches related to input: ")
            for num, it in enumerate(cnf_dis):
                st.write(num, ")", it)
            if num != 0:
                conf_inp = st.number_input(f"Select the one you meant (0 - {num}):", min_value=0, max_value=num)
                disease_input = cnf_dis[conf_inp]
                break
            else:
                disease_input = cnf_dis[0]
                break
        else:
            st.write("Enter a valid symptom.")

    while True:
        try:
            num_days = int(st.text_input("Okay. From how many days? : "))
            break
        except:
            st.write("Enter a valid input.")

    def recurse(node, depth):
        if tree_.feature[node] != _tree.TREE_UNDEFINED:
            name = feature_name[node]
            threshold = tree_.threshold[node]

            if name == disease_input:
                val = 1
            else:
                val = 0
            if val <= threshold:
                recurse(tree_.children_left[node], depth + 1)
            else:
                symptoms_present.append(name)
                recurse(tree_.children_right[node], depth + 1)
        else:
            present_disease = print_disease(tree_.value[node])
            red_cols = reduced_data.columns
            symptoms_given = red_cols[reduced_data.loc[present_disease].values[0].nonzero()]
            st.write("Are you experiencing any ")
            symptoms_exp = []
            for syms in list(symptoms_given):
                inp = ""
                while True:
                    inp = st.radio(f"{syms}?", ("yes", "no"))
                    if inp == "yes" or inp == "no":
                        break
                    else:
                        st.write("Please provide a valid answer (yes/no).")

                if inp == "yes":
                    symptoms_exp.append(syms)

            second_prediction = sec_predict(symptoms_exp)
            calc_condition(symptoms_exp, num_days)
            if present_disease[0] == second_prediction[0]:
                st.write("You may have ", present_disease[0])
                st.write(description_list.get(present_disease[0], "Description not available."))

            else:
                st.write("You may have ", present_disease[0], "or ", second_prediction[0])
                st.write(description_list.get(present_disease[0], "Description not available."))
                st.write(description_list.get(second_prediction[0], "Description not available."))

            precution_list = precautionDictionary.get(present_disease[0], [])
            if precution_list:
                st.write("Take the following measures:")
                for i, j in enumerate(precution_list, 1):
                    st.write(f"{i}) {j}")
            else:
                st.write("Precautionary measures not available for this condition.")

    recurse(0, 1)

getSeverityDict()
getDescription()
getprecautionDict()
getInfo()
tree_to_code(clf, cols)
st.write("----------------------------------------------------------------------------------------")
