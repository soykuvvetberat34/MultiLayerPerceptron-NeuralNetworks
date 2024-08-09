from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split,GridSearchCV
import pandas as pd
import numpy as np

df_=pd.read_csv("diabetes.csv")
y=df_["Outcome"]
df=df_.drop(["Outcome"],axis=1)
x_train,x_test,y_train,y_test=train_test_split(df,y,test_size=0.3,random_state=200)

MLPCmodel=MLPClassifier()
MLPC_params={
    "alpha":[0.0001,0.00001,0.000001,0.001,0.01],
    "hidden_layer_sizes":[(50,50),(100,100),(20,20)],
}

MLPC_cv=GridSearchCV(MLPCmodel,MLPC_params,cv=5,n_jobs=-1,verbose=2)
MLPC_cv.fit(x_train,y_train)
alpha=MLPC_cv.best_params_["alpha"]
hidden_layer_sizes=MLPC_cv.best_params_["hidden_layer_sizes"]



MLPC_tuned=MLPClassifier(hidden_layer_sizes=hidden_layer_sizes,alpha=alpha)
MLPC_tuned.fit(x_train,y_train)
predict=MLPC_tuned.predict(x_test)
accuracy_score=accuracy_score(y_test,predict)
print(accuracy_score)

