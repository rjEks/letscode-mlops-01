#!/usr/bin/env python2
# -*- coding: utf-8 -*-

from sklearn import svm, datasets
import joblib 
import numpy as np


iris = datasets.load_iris()
X = iris.data  
y = iris.target

model = svm.SVC(kernel='poly', degree=3, C=1.0).fit(X, y)

joblib.dump(model, "modelo.joblib")