# 忽略警告信息
import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt
import copy
import shap
import lightgbm as lgb
import sklearn.preprocessing 
import sklearn.feature_selection
import sklearn.metrics as metrics
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from scipy.stats.stats import spearmanr, pearsonr
import sklearn.model_selection as ms
import sklearn.svm as svm
import joblib
def evaluate_performance(y_test, y_pred, y_prob):
    # AUROC
    auroc = metrics.roc_auc_score(y_test,y_prob)
    auroc_curve = metrics.roc_curve(y_test, y_prob)
    # AUPRC
    auprc=metrics.average_precision_score(y_test, y_prob) 
    auprc_curve=metrics.precision_recall_curve(y_test, y_prob)
    #Accuracy
    accuracy=metrics.accuracy_score(y_test,y_pred) 

    recall=metrics.recall_score(y_test, y_pred)
    precision=metrics.precision_score(y_test, y_pred)
    f1=metrics.f1_score(y_test, y_pred)
    class_report=metrics.classification_report(y_test, y_pred,target_names = ["control","case"])

    model_perf = {"auroc":auroc,"auroc_curve":auroc_curve,
                  "auprc":auprc,"auprc_curve":auprc_curve,
                  "accuracy":accuracy,
                  "recall":recall,"precision":precision,"f1":f1,
                  "class_report":class_report}
        
    return model_perf

# Output result of evaluation
#生成文件可以用于作图（bar plot consisted of accuracy,sensitivity,specificity,auroc,f1 score,precision,recall,auprc ）
def eval_output(model_perf,output_file):
    with open(output_file,'w') as f:
        f.write("AUROC=%s\tAUPRC=%s\tAccuracy=%s\tRecall=%s\tPrecision=%s\tf1_score=%s\n" %
               (model_perf["auroc"],model_perf["auprc"],model_perf["accuracy"],model_perf["recall"],model_perf["precision"],model_perf["f1"]))
        f.write("\n######NOTE#######\n")
        f.write("#According to help_documentation of sklearn.metrics.classification_report:in binary classification, recall of the positive class is also known as sensitivity; recall of the negative class is specificity#\n\n")
        f.write(model_perf["class_report"])

        
# Plot AUROC of model
def plot_AUROC(model_perf,output_file):
    #get AUROC,FPR,TPR and threshold
    roc_auc = model_perf["auroc"]
    fpr,tpr,threshold = model_perf["auroc_curve"]
    #plot
    plt.figure()
    lw = 2
    plt.figure(figsize=(10,10))
    plt.plot(fpr, tpr, color='darkorange',
             lw=lw, label='AUROC (area = %0.2f)' % roc_auc) #%相当于format方法,格式化输出,替换%***
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC of Models")
    plt.legend(loc="lower right")
    plt.savefig(output_file,format = "pdf")
    
    
# Plot AUPRC of model
def plot_AUPRC(model_perf,output_file):
    #get AUPRC,Precision,Recall and threshold
    prc_auc = model_perf["auprc"]
    precision,recall,threshold = model_perf["auprc_curve"]
    #plot
    plt.figure()
    lw = 2
    plt.figure(figsize=(10,10))#设置figure的宽、长
    plt.plot(recall, precision, color="red",
             lw=lw,#折线图的线条宽度
             label='AUPRC (area = %0.2f)' % prc_auc#图例
            ) 
    plt.plot([0, 1], [1, 0], color="navy", lw=lw, linestyle='--')#对角线
    #设置x、y坐标轴的范围
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    #坐标轴标签
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("AUPRC of Models")
    plt.legend(loc="lower right")
    plt.savefig(output_file,format = "pdf")

    
    
#Random seed
SEED = np.random.seed(2020)
############ Data Processing ###########
print("\n...... Processing Data ......\n")

#Load data
GTEx = pd.read_csv('/home/lilabguest2/shenhaoyu/GTEx_ml/GTEx_dataset_mult_outlier',sep = '\t')#,index_col=0
GTEx_feature=GTEx.drop(['SubjectID','GeneName','eoutlier','over_eoutlier','under_eoutlier','aseoutlier','spliceoutlier','outlier'],axis=1)
out=GTEx['spliceoutlier']
#Input
X = copy.deepcopy(GTEx_feature)#经过deepcopy的操作的不管是内层还是外层数据对象都拥有不同的地址空间，修改其中的值不会对两个对象都造成影响
y = copy.deepcopy(out)
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.2,random_state = SEED)
print("\n...... Finished Data Processing ......\n")
########################################
########## Model Construction ##########
print("\n...... Training Model ......\n")

# SVM params
param_dict = {
    "kernel": ['poly','rbf','sigmoid'],
    "gamma": [1, 0.1, 0.01],
    "C":[0.1, 1, 10],
    "random_state":[2020],#
    'probability':[True]
}#参数列表

#Initiate model
model = svm.SVC()#创建一个分类器对象
#Adjust hyper-parameters with 5-fold cross validation
rscv = RandomizedSearchCV(model,#
                          param_dict,#
                          cv = 5,# Determines the cross-validation splitting strategy.
                          verbose = 0,# Controls the verbosity: the higher, the more messages.
                          scoring = "roc_auc",#
                          n_jobs =-1#
                         )#用分类器对象创建一个RandomizedSearchCV的对象
gbm=rscv.fit(X_train, y_train)#得到训练后的模型 
########## Model Evaluation ##########
print("\n...... Evaluating Model ......\n")
#Get best model with score [max(mean(auc(5 cross validation)))]
best_model = rscv.best_estimator_
#Get predict_class(y_pred) and predict_probality_for_case(y_prob) of TestSet
y_pred = best_model.predict(X_test)
y_prob = best_model.predict_proba(X_test)[:,1]

#绘制学习曲线
from sklearn.model_selection import learning_curve
from sklearn.model_selection import ShuffleSplit
cv = ShuffleSplit(n_splits=10, test_size=0.2, random_state=0)
train_sizes, train_scores, CV_scores=learning_curve(best_model,X_train,y_train,cv=cv,train_sizes=np.linspace(.1, 1.0, 6),scoring='neg_mean_squared_error',n_jobs=-1)
train_scores_mean = np.mean(train_scores, axis=1)
train_scores_std = np.std(train_scores, axis=1)
CV_scores_mean = np.mean(CV_scores, axis=1)
CV_scores_std = np.std(CV_scores, axis=1)
plt.grid()
plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
plt.fill_between(train_sizes, CV_scores_mean - CV_scores_std,
                     CV_scores_mean + CV_scores_std, alpha=0.1, color="g")
plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
plt.plot(train_sizes, CV_scores_mean, 'o-', color="g",
             label="Cross-validation score")
plt.legend(loc="best")
plt.savefig('./learning_curve_svm.pdf',format = "pdf")
#Get model performance
model_perf = evaluate_performance(y_test,y_pred,y_prob)
#Output result of evaluation
eval_output(model_perf,"./Evaluate_Result_TestSet_binerary_splice_svm.txt")
#You can make bar plot consisted of accuracy,sensitivity,specificity,auroc,f1 score,precision,recall,auprc according to the "Evaluate_Result_TestSet.txt"
#Plot
# plot AUROC AUPRC
plot_AUROC(model_perf,"./AUROC_TestSet_binerary_splice_svm.pdf")
plot_AUPRC(model_perf,"./AUPRC_TestSet_binerary_splice_svm.pdf")

print("\n...... Finished Model Evaluation ......\n")
######################################