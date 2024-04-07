from sklearn.metrics import confusion_matrix,precision_recall_fscore_support,accuracy_score,cohen_kappa_score
from sklearn import metrics
import matplotlib.pyplot as plt
import numpy as np


def evaluation_metric(y_true, y_pre,n_classes=2):
    """
    y_true: true label
    y_pre: predict label
    n_classes: class number
    
    evaluation_metric
    
    ## binary: AUC ACC F1 P R MACC BACC

    ## multi label: QWK AUC ACC F1 P R MACC BACC

    ## regression: R R2 MAE RMSE top10% top20% bottom10% bottom20%

    ## ranking: NDCG N% top10% top20% bottom10% bottom20%
    
    """
    result = dict()
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    
    acc = accuracy_score(y_true, y_pre)
    qwk = cohen_kappa_score(y_true, y_pre,weights='quadratic')
    pre_avg, recall_avg, f1_avg, sur = precision_recall_fscore_support(y_true, y_pre, average='macro')
    
    result['ACC'] = round(acc,2)
    result['F1'] = round(f1_avg,2)
    result['QWK'] = round(qwk,2)
    result['Precision'] = round(pre_avg,2)
    result['Recall'] = round(recall_avg,2)
    
    if n_classes==2:
        try:
            auc = metrics.roc_auc_score(y_true, y_pre)
        except:
            auc = 0
        result['AUC'] = round(auc,2)
    else:
        result['AUC'] = 0
        
    return result

def Draw_plot(y_test,y_pred,Fig_title = '',x_title = 'Measured',y_title = 'Predicted',x_axis_mv=8.5,y_axis_mv=0.7,fold4 = 0.6 ,fold8 = 0.778,xmin='', xmax='', ymin='', ymax =''):

    measured = list(y_test)
    predicted = list(y_pred)
    correlation_matrix = np.corrcoef(measured, predicted)
    correlation_xy = correlation_matrix[0,1]
    R_score = correlation_xy
    performance = 0
    mse_per = 0
    for i in range(len(measured)):
        performance += abs(predicted[i] - measured[i])
        mse_per += pow((predicted[i] - measured[i]), 2)

    performance = performance / len(measured)
    mse_per = mse_per / len(measured)
    plt.xlabel(x_title, fontsize=12)
    plt.ylabel(y_title, fontsize=12)
    z = np.polyfit(measured, predicted, 1)
    p = np.poly1d(z)

    plt.scatter(measured,predicted,s=4,color='b')

    xdata_range = np.linspace(int(np.floor(xmin)),int(np.ceil(xmax)),7)
    ydata_range = np.linspace(int(np.floor(ymin)),int(np.ceil(ymax)),7)
    plt.xticks(xdata_range)
    plt.yticks(ydata_range)
    plt.plot([xmin, xmax], [ymin, ymax], color='grey', linestyle='solid')
    plt.title(Fig_title)


    plt.text((xmin+x_axis_mv),(ymax-y_axis_mv), "%s = %.3f" % ('MAE', performance),
             color='black', fontsize=10, style='normal',
             # bbox={'facecolor':'red', 'alpha':0.5},
             horizontalalignment='left')
    plt.text((xmin+x_axis_mv), (ymax-y_axis_mv*2) , "%s = %.3f" % ('RMSE', np.sqrt(mse_per)),
             color='black', fontsize=10, style='normal',
             # bbox={'facecolor':'red', 'alpha':0.5},
             horizontalalignment='left')
    plt.text((xmin+x_axis_mv), (ymax-y_axis_mv*3), "{} = {}".format('R', round(R_score, 3)),
             color='black', fontsize=10, style='normal')

    plt.gca().set_aspect('equal', 'box')
    plt.show()


def draw_confusion(y_true, y_pre,labels,model=''):
    """
    draw classification confusion figure

    y_true: true label
    y_pre: predict label
    labels: category label
    
    """
    
    cm = confusion_matrix(y_pre,y_true)
    
    if cm.size == 1:
        flag = 1
        value = cm[0][0]
        if y_true[0] == 0:
            cm = np.array([[value,0],[0,0]])
        elif y_true[0] == 1:
            cm = np.array([[0,0],[0,value]])
        else:
            print("All Prediction False or You should input counts value!")
    else:
        flag = 0
        
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(cm, cmap=plt.cm.Oranges)
    fig.colorbar(cax)

    result = evaluation_metric(y_true, y_pre,n_classes=len(labels))

    indices = range(len(labels))    
    plt.title(f'{model} vs. exp Confusion Matrix')

    iters = np.reshape([[[i, j] for j in range(len(labels))] for i in range(len(labels))], (cm.size, 2))
    for i, j in iters:
        plt.text(j, i, format(cm[i, j])) 
    
    plt.xticks(indices, labels) 
    plt.yticks(indices, labels)
    y_mv=0.1*len(labels)
    x_mv = 0.55*len(labels)

    plt.text(x_mv*2.2,y_mv*0, "%s = %.3f" % (f'ACC',result['ACC']),
             color='black', fontsize=10, style='normal',bbox={'facecolor':'lightskyblue', 'alpha':0.1},
             horizontalalignment='left')
    plt.text(x_mv*2.2,y_mv*1, "%s = %.3f" % (f'F1',result['F1']),
             color='black', fontsize=10, style='normal',bbox={'facecolor':'lightskyblue', 'alpha':0.1},
             horizontalalignment='left')
    plt.text(x_mv*2.2,y_mv*2, "%s = %.3f" % (f'Precision',result['Precision']),
             color='black', fontsize=10, style='normal',bbox={'facecolor':'lightskyblue', 'alpha':0.1},
             horizontalalignment='left')
    plt.text(x_mv*2.2,y_mv*3, "%s = %.3f" % (f'Recall',result['Recall']),
             color='black', fontsize=10, style='normal',bbox={'facecolor':'lightskyblue', 'alpha':0.1},
             horizontalalignment='left')


    plt.ylabel('\n Experiment 1\n')
    plt.xlabel('\n Experiment 2\n')
    plt.tight_layout()
    plt.axis('off')
    plt.show()
    
    if (flag == 1):
        if y_true[0] == 0:
            labels = ['0-10']
        else:
            labels = ['>=10']        