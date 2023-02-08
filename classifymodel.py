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
from sklearn import tree
from sklearn.multiclass import OneVsRestClassifier
from sklearn.utils.class_weight import  compute_sample_weight
from sklearn.svm import SVC
from itertools import cycle
from numpy import interp
from sklearn.preprocessing import label_binarize

inputfile = r"D:\study\data\chenan\2022\Tellurene\ML\data\tell_bandgap.csv"
def main():
    df = pd.read_csv(inputfile, encoding='gbk')
    df.drop(['system'],1,inplace=True)
    #train = shuffle(train, random_state=1)
    #print(train)
    x = np.array(df.drop(['bandgap'],1))
    y1 = np.array(df['type'])
    #target2 = np.array(df['tunnling_probability'])
    y_s = list()
    #target_t = list()
    for tp in y1:
        if tp == 'RP':
            newclass1 = 0
            y_s.append(newclass1)
        elif tp == 'DJ':
            newclass1 = 1
            y_s.append(newclass1)
        else:
            newclass1 = 2
            y_s.append(newclass1)
    y_s = label_binarize(y_s, classes=[0, 1, 2])
    n_classes = y_s.shape[1]

    training_x, testing_x, training_y, testing_y = \
                train_test_split(x, y_s, test_size=0.2)
    #print(training_x.shape)
    #print(training_y.shape)
    random_state = np.random.RandomState(0)
    model = OneVsRestClassifier(SVC(kernel='linear', probability=True,
                                 random_state=random_state))

    y_score=model.fit(training_x, training_y)
    y_pred = model.predict(testing_x)
    print('results：')
    print(y_pred)
    #print(y_pred_p)
    accuracy = accuracy_score(testing_y, y_pred)
    print("accuracy_score:")
    print(accuracy)

# 计算每一类的ROC
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(testing_y[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

# Compute micro-average ROC curve and ROC area（方法二）
    fpr["micro"], tpr["micro"], _ = roc_curve(testing_y.ravel(), y_score.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    aucsocre = roc_auc_score(testing_y, y_pred)
    print("auc_score:" )
    print(aucsocre)
# Compute macro-average ROC curve and ROC area（方法一）
# First aggregate all false positive rates
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
# Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += interp(all_fpr, fpr[i], tpr[i])
# Finally average it and compute AUC
    mean_tpr /= n_classes
    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])
# Plot all ROC curves
    lw=2
    plt.figure()
    plt.plot(fpr["micro"], tpr["micro"],
            label='micro-average ROC curve (area = {0:0.2f})'
                ''.format(roc_auc["micro"]),
            color='deeppink', linestyle=':', linewidth=4)

    plt.plot(fpr["macro"], tpr["macro"],
            label='macro-average ROC curve (area = {0:0.2f})'
                ''.format(roc_auc["macro"]),
            color='navy', linestyle=':', linewidth=4)

    colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
    for i, color in zip(range(n_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=lw,
                label='ROC curve of class {0} (area = {1:0.2f})'
                ''.format(i, roc_auc[i]))

    plt.plot([0, 1], [0, 1], 'k--', lw=lw)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Some extension of Receiver operating characteristic to multi-class')
    plt.legend(loc="lower right")
    plt.show()
#混淆矩阵
    sw = compute_sample_weight(class_weight='balanced', y=testing_y)
    cfm = confusion_matrix(testing_y, y_pred, sample_weight=sw)
    print(cfm)
    colormap=plt.cm.Blues
    plt.matshow(cfm, cmap=colormap)
    plt.show()

'''
    kf = KFold(n_splits=5, shuffle=True, random_state=1)
    ### WZL ###
    # i = 0
    # # for train, test in kf.split(x, y_s):
    # #     i = i + 1
    #     x = np.array(x)
    #     y_s = np.array(y_s)
    #    print(np.array(x)[train])
    #    model = xc1(colsample_bytree=0.81, learning_rate=0.01,
    #    max_depth=5,  n_estimators=140, nthreaad=1, subsample=0.8, scale_pos_weight=4,
    #    min_child_weight=1, gamma=1)
    #    model.fit(x[train], y_s[train])
    #    fpr, tpr, thres = roc_curve(y_s[test], model.predict(x[test]))
    #    results = np.vstack([fpr, tpr, thres])
     #   np.savetxt("results"+str(i)+".txt", results.T)
    #########
    cvs = cross_val_score(model, x, y_s, cv=kf, scoring= 'accuracy')
    print(cvs)
    print("cvs_mean:")
    print(cvs.mean())
'''
    #保存模型
    #joblib.dump(model,'0910xgbrclassiof0723Z140TB.pkl')

    #fig = plot_roc_curve(model, testing_x, testing_y)
    #plt.savefig("D:\study\data\chenan\\2020\ML\project\hetero\ROC.png")
'''
    #AUC曲线
    tprs = []
    aucs = []
    mean_fpr = np.linspace(0,1,100)

    fig, ax = plt.subplots()
    for i, (train, test) in enumerate(kf.split(x, y_s)):
        x = np.array(x)
        y_s = np.array(y_s)
        model.fit(x[train],y_s[train])
        viz = plot_roc_curve(model, x[test],y_s[test],
                             name='Fold-{}'.format(i+1),
                             alpha=0.4, lw=1, ax=ax)
        interp_tpr = np.interp(mean_fpr, viz.fpr, viz.tpr)
        interp_tpr[0]=0.0
        tprs.append(interp_tpr)
        aucs.append(viz.roc_auc)

    ax.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
        label='', alpha=.8)

    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)
    ax.plot(mean_fpr, mean_tpr, color='dodgerblue',
            label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),
            lw=2, alpha=1)

    std_tpr = np.std(tprs, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    ax.fill_between(mean_fpr, tprs_lower, tprs_upper, color='green', alpha=.2,
                label=r'$\pm$ 1 std. dev.')

    ax.set(xlim=[-0.0, 1.0], ylim=[-0.0, 1.05],
       title="Receiver operating characteristic")
    ax.legend(loc="lower right", shadow=True, fontsize=7.4, fancybox=True)
'''


if __name__ == "__main__":
    main()
