import numpy as np
import pandas as pd
from IPython.display import display
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split,StratifiedKFold, GridSearchCV
from sklearn.datasets import load_iris
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

plt.rcParams['font.family'] = 'AppleGothic'
plt.rcParams['axes.unicode_minus'] = False
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)

# 데이터 샘플확인
iris = load_iris()

X = pd.DataFrame(
    iris.data,
    columns=iris.feature_names
) 
y = iris.target
target_names = iris.target_names
print(X.head())

#데이터 분리
X_train, X_test, y_train, y_test = train_test_split(
    X, y,test_size=0.2,random_state=42,stratify=y)

print(f"Train: {X_train.shape}")


#파이프라인
pipe_li = Pipeline([
    ('clf',RandomForestClassifier(random_state=42))
])

print("Pipeline 구성완료")

#하이퍼파라미터 그리드
param_grid = {
    'clf__n_estimators' : [10,50,100,200],
    'clf__max_depth' : [None, 10, 20],
    'clf__min_samples_split' : [2,5],
    'clf__min_samples_leaf' : [1, 2]
}

# GridSearchCV 생성

grid_clf = GridSearchCV(
    pipe_li,
    param_grid,
    cv=StratifiedKFold(5),
    scoring='accuracy',
    n_jobs=-1,
    verbose=1
)

# fitting 단계
grid_clf.fit(X_train, y_train)
print('GridSearch 완료!')


#예측,성능확인
y_pred = grid_clf.best_estimator_.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion_matrix:")
print(confusion_matrix(y_test, y_pred))

#사람기준 꽃이름 결과확인
predicted_flowers = iris.target_names[y_pred]
true_flowers = iris.target_names[y_test]

print(predicted_flowers[:5])
print(true_flowers[:5])

# 새꽃 입력/예측
new_flower = [[5.1, 3.5, 1.4, 0.2]]

pred = grid_clf.best_estimator_.predict(new_flower)
predicted_flowers_name = target_names[pred][0]

print("예측된 꽃:", predicted_flowers_name)
