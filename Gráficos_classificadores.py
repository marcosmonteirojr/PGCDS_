import matplotlib.pyplot as plt
import numpy as np
import Cpx
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

# Importing DS techniques
from deslib.dcs.ola import OLA
from deslib.dcs.rank import Rank
from deslib.des.des_p import DESP
from deslib.des.knora_e import KNORAE
from deslib.static import StackedClassifier
from deslib.util.datasets import make_P2


# Plotting-related functions
def make_grid(x, y, h=.01):
    x_min, x_max = x.min() - 1, x.max() + 1
    y_min, y_max = y.min() - 1, y.max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    return xx, yy


def plot_classifier_decision(ax, clf, X, mode='line', **params):
    xx, yy = make_grid(X[:, 0], X[:, 1])

    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    if mode == 'line':
        ax.contour(xx, yy, Z,linewidths=0.3,colors="red")
    else:
        ax.contourf(xx, yy, Z, **params)
    ax.set_xlim((np.min(X[:, 0]), np.max(X[:, 0])))
    ax.set_ylim((np.min(X[:, 1]), np.max(X[:, 0])))


def plot_dataset(X, y, ax=None, title=None, **params):
    if ax is None:
        ax = plt.gca()
    ax.scatter(X[:, 0], X[:, 1], marker='o', c=y, s=25,
               edgecolor='k',**params)
    ax.set_xlabel('Atributo 1')
    ax.set_ylabel('Atributo 2')
    if title is not None:
        ax.set_title(title)
    return ax

local_dataset = "/media/marcos/Data/Tese/Bases3/Dataset/"
local = "/media/marcos/Data/Tese/Bases3/"
cpx_caminho="/media/marcos/Data/Tese/Bases3/Bags/"
base_name="P2"

X,y,_,_=Cpx.open_data(base_name,local_dataset)

Treino=Cpx.open_training(local,base_name,1)



bags_ga="distdiverlinear"

bags=Cpx.open_bag(cpx_caminho+"1/",base_name+"")
bags2 = Cpx.open_bag(cpx_caminho+str(1)+"/", base_name + bags_ga)
val,test=Cpx.open_test_vali(local,base_name,"1")
X_t,y_t=Cpx.biuld_x_y(Treino,X,y)
Xval,y_val=Cpx.biuld_x_y(val,X,y)
Xtest,y_test=Cpx.biuld_x_y(test,X,y)


#X_t=Cpx.min_max_norm(X_t)
#Xval=Cpx.min_max_norm(Xval)
#Xtest=Cpx.min_max_norm(Xtest)

Xval=np.array(Xval)
y_val=np.array(y_val)

Xtest=np.array(Xtest)
y_test=np.array(y_test)


X=np.array(X)
y=np.array(y)

X_t=np.array(X_t)
y_t=np.array(y_t)

classifier=[]
classifier2=[]
#print(len(bags['inst']))
#for i in range(10):
#print(i)
for i in range(1,99):
    #bags2 = Cpx.open_bag(cpx_caminho + str(1) + "/", base_name )
    X_b, y_b = Cpx.biuld_x_y(bags['inst'][i], X, y)
   # X_b=Cpx.min_max_norm(X_b)
    X_b=np.array(X_b)
    y_b=np.array(y_b)
    ca, sc, _ = Cpx.biuld_classifier(X_b, y_b, Xval, y_val)
    classifier.append(ca)

for i in range(1,99):
    #bags2 = Cpx.open_bag(cpx_caminho + str(1) + "/", base_name )
    X_b2, y_b2 = Cpx.biuld_x_y(bags2['inst'][i], X, y)
   # X_b=Cpx.min_max_norm(X_b)
    X_b2=np.array(X_b2)
    y_b2=np.array(y_b2)
    ca2, sc2, _ = Cpx.biuld_classifier(X_b2, y_b2, Xval, y_val)
    classifier2.append(ca2)



for clf in classifier:
    #print(sc)
    ax = plot_dataset(Xval, y_val)

    plot_classifier_decision(ax, clf, Xval)

    ax.set_xlim((0, 1))
    ax.set_ylim((0, 1))

    #plt.savefig('Bag'+str(i)+'+.png', dpi=300)
    #ax.clear()
plt.show()
for clf in classifier2:
    ax2 = plot_dataset(Xval, y_val)
    plot_classifier_decision(ax2, clf, Xval)
    ax2.set_xlim((0, 1))
    ax2.set_ylim((0, 1))

plt.show()
    #plt.tight_layout()