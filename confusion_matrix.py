import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score, KFold, LeaveOneOut
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from xgboost.sklearn import XGBClassifier as xc1
from sklearn.metrics import accuracy_score, precision_score, plot_roc_curve, roc_curve, roc_auc_score,auc
from sklearn.utils import shuffle
from sklearn.model_selection import GridSearchCV
import joblib
from sklearn.metrics import confusion_matrix

inputfile = r"D:\study\data\chenan\2022\Tellurene\ML\data\tell_bandgap.csv"
def main():
    df = pd.read_csv(inputfile)
    df.drop(['system'],1,inplace=True)
    x = np.array(df.drop(['bandgap'],1))
    y1 = np.array(df['bandgap'])
    #target2 = np.array(df['tunnling_probability'])
    y_s = list()
    #target_t = list()
    for newclass1 in y1:
        if 0.26 < newclass1:
            newclass1 = 1
            y_s.append(newclass1)
        else:
            newclass1 = 0
            y_s.append(newclass1)



    training_x, testing_x, training_y, testing_y = \
                train_test_split(x, y_s, test_size=0.2,random_state=5)

    model = xc1(colsample_bytree=0.5, learning_rate=0.01,
                max_depth=6,  n_estimators=120, nthreaad=1, subsample=0.6)

    model.fit(training_x, training_y)
    y_pred = model.predict(testing_x)
    print('results：')
    print(y_pred)
    #print(y_pred_p)
    accuracy = accuracy_score(testing_y, y_pred)
    print("accuracy_score:")
    print(accuracy)
    aucsocre = roc_auc_score(testing_y, y_pred)
    print("auc_score:" )
    print(aucsocre)


    kf = KFold(n_splits=5, shuffle=True, random_state=1)
    cvs = cross_val_score(model, training_x, training_y, cv=kf, scoring= 'accuracy')
    print(cvs)
    print("cvs_mean:")
    print(cvs.mean())

    #混淆矩阵
    cfm = confusion_matrix(testing_y, y_pred)
    print(cfm)
    colormap=plt.cm.Blues
    plt.matshow(cfm, cmap=colormap)

    plt.show()


if __name__ == "__main__":
    main()
