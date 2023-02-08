import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error,mean_absolute_error
import matplotlib.pyplot as plt
import joblib
from xgboost.sklearn import XGBRegressor as xgbr
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from scipy.optimize import curve_fit
from sklearn.preprocessing import StandardScaler as SS
from sklearn.preprocessing import StandardScaler, MinMaxScaler, PolynomialFeatures
from xgboost import XGBClassifier as xc1
from warnings import simplefilter


input1= r"D:\study\data\chenan\2020\ML\project\hetero\220410\traningdata_0427_137.csv"
simplefilter(action='ignore', category=FutureWarning)
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
                train_test_split(x, y_s, test_size=0.2, random_state=5)

    model = xc1(colsample_bytree=0.5, learning_rate=0.01,
                max_depth=6,  n_estimators=120, nthreaad=1, subsample=0.6)
    model.fit(training_x, training_y)

    feature_importance = model.feature_importances_
    #make importances relative to max importance
    feature_importance = 100.0*(feature_importance/feature_importance.max())
    sorted_idx=np.argsort(feature_importance)
    pos = np.arange(sorted_idx.shape[0])+.5
    plt.subplot(1,2,2)
    plt.barh(pos,feature_importance[sorted_idx], align='center')
    labels=list(df.columns.values)
    feature={'feature':[labels[sorted_idx[i]] for i in range(len(sorted_idx))],'Relative Importance':feature_importance[sorted_idx]}
    #S_f=pd.DataFrame(feature)
    #S_f.to_csv("feature.csv")
    #trainingdata={'real':train_target,'predict':exported_pipeline.predict(train_data)}
    #GOH2=pd.DataFrame(trainingdata)
    #GOH2.to_csv("training.csv")
    testingdata={'real':testing_y,'predict':model.predict(testing_x)}

    plt.yticks(pos,[labels[sorted_idx[i]] for i in range(len(sorted_idx))])
    plt.xlabel('Relative Importance')
    plt.title('Variable Importance')

    plt.plot()


    plt.show()


if __name__ == "__main__":
    main()
