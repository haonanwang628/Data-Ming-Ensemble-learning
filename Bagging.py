#集合分类器构建命名
print("WHN集合分类器运行")
# 根据数据导入各种模块
import numpy as np
import csv
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.externals import joblib #j bolib模块
from sklearn.model_selection import KFold # K折交叉验证模块
from collections import Counter
from sklearn.metrics import matthews_corrcoef
from sklearn import tree # 决策树算法模块
from sklearn.neighbors import KNeighborsClassifier # KNN算法模块
from sklearn.naive_bayes import GaussianNB # 朴素贝叶斯模块中的高斯贝叶斯分类器
from sklearn.neural_network import MLPClassifier # BP神经网络模块
print("WHN集合分类器代入模块成功")
print("WHN集合分类器随机采样生成子数据集....")
# 随机采样生成子数据集
def Sampling(X_train, Y_train):
    m, n = np.shape(X_train)
    index = np.random.randint(m, size=m)
    subSet_X = X_train[index, :]
    subSet_Y = Y_train[index]
    choosed = np.unique(index)
    un_choosed = np.delete(range(m),choosed)
    out_of_beg_X = np.delete(X_train, choosed, axis=0)  # 加密货币价格数据集
    out_of_beg_Y = np.delete(Y_train, choosed, axis=0)  # 加密货币价格数据集
    return subSet_X, subSet_Y, out_of_beg_X, out_of_beg_Y, un_choosed
print("WHN集合分类器Bagging方法，正在分别定义子采样器，分别选择决策树，KNN，朴素贝叶斯，BP神经网络为个体学习器")
# 定义Bagging集成学习器，分别选择决策树，KNN，朴素贝叶斯，BP神经网络为个体学习器
print("[0-25]WHN集合分类器子学习器此时是——决策树...")
print("[26-50]WHN集合分类器子学习器此时是——KNN近邻算法...")
print("[51-75]WHN集合分类器子学习器此时是——朴素贝叶斯...")
print("[76-100]WHN集合分类器子学习器此时是——BP神经网络...")
def Bagging(X_train, X_test, Y_train, Y_test):
    model = []
    m = len(X_train)
    Num = 100 #子学习器个数
    testGallary = []
    e_knn = [];e_tree = [];e_gnb = [];e_bp = [];#定义列表，存放个体学习器错误率
    sumList = [[] for i in range(m)]
    for t in np.arange(Num):
        if t < 25:
            # 子学习器是决策树
            Tree = tree.DecisionTreeClassifier()
            subSet_X, subSet_Y, out_of_beg_X, out_of_beg_Y, un_choosed = Sampling(X_train, Y_train)
            Tree.fit(subSet_X, subSet_Y)
            y = Tree.predict(out_of_beg_X)
            testGallary.append(Tree.predict(X_test))
            model.append(Tree)
            # 决策树错误率计算
            Tree.fit(X_train, Y_train)
            e_tree.append(1 - Tree.score(X_train, Y_train))

            # 记录预测结果
            i = 0
            for item in un_choosed:
                sumList[item].append(y[i])
                i += 1
        elif t < 50:
                # 子学习器是KNN
                KNN = KNeighborsClassifier()
                subSet_X, subSet_Y, out_of_beg_X, out_of_beg_Y, un_choosed = Sampling(X_train, Y_train)
                KNN.fit(subSet_X, subSet_Y)
                y = KNN.predict(out_of_beg_X)
                testGallary.append(KNN.predict(X_test))
                model.append(KNN)
                # KNN错误率计算
                KNN.fit(X_train, Y_train)
                e_knn.append(1 - KNN.score(X_train, Y_train))
                # 记录预测结果
                i = 0
                for item in un_choosed:
                    sumList[item].append(y[i])
                    i += 1
        elif t < 75:
            # 子学习器是朴素贝叶斯
            GNB = GaussianNB()
            subSet_X, subSet_Y, out_of_beg_X, out_of_beg_Y, un_choosed = Sampling(X_train, Y_train)
            GNB.fit(subSet_X, subSet_Y)
            y = GNB.predict(out_of_beg_X)
            testGallary.append(GNB.predict(X_test))
            model.append(GNB)
            # 朴素贝叶斯错误率计算
            GNB.fit(X_train, Y_train)
            e_gnb.append(1 - GNB.score(X_train, Y_train))
            # 记录预测结果
            i = 0
            for item in un_choosed:
                sumList[item].append(y[i])
                i += 1

        else:
            # 子学习器是BP神经网络
            BP = MLPClassifier()
            subSet_X, subSet_Y, out_of_beg_X, out_of_beg_Y, un_choosed = Sampling(X_train, Y_train)
            BP.fit(subSet_X, subSet_Y)
            y = BP.predict(out_of_beg_X)
            testGallary.append(BP.predict(X_test))
            model.append(BP)
            # BP神经网络错误率计算
            BP.fit(X_train, Y_train)
            e_bp.append(1 - BP.score(X_train, Y_train))

            # 记录预测结果
            i = 0
            for item in un_choosed:
                sumList[item].append(y[i])
                i += 1

    # 计算集成学习器错误率
    index = []
    for i in range(len(sumList)):
        if sumList[i] != []:
            index.append(i)
    oob = []
    for i in index:
        dic_oob = Counter(sumList[i]).most_common(1)
        oob.append(dic_oob[0][0])
    oob_y = Y_train[index]
    accr_oob = 0
    for item in np.arange(len(oob_y)):
        if oob[item] == oob_y[item]:
            accr_oob += 1
    err = 1 - accr_oob / m
    print('集成学习器错误率：')
    print(err)

    # 计算多样性评价指标（相关系数）
    corr_coefficient = np.empty([Num, Num])
    for i in range(Num):
        for j in range(Num):
            corr_coefficient[i, j] = matthews_corrcoef(testGallary[i], testGallary[j])

    # 个体学习器错误率
    errknn = np.max(e_knn)
    errtree = np.max(e_tree)
    errgnb = np.max(e_gnb)
    errbp = np.max(e_bp)

    return err, corr_coefficient, testGallary, model, errknn, errtree, errgnb, errbp

# 集成学习
# 读取加密货币价格数据集
csv_reader = csv.reader(open('new加密货币价格数据.csv','r'))
dataSet = []
for line in csv_reader:
    dataSet.append(list(map(float,line)))
DataSet = np.array(dataSet)
m,n = np.shape(DataSet)
htru_X = DataSet[:, 0:n-1]
htru_Y = DataSet[:, n-1]

# 划分训练集和测试集
X_train, X_test, Y_train, Y_test = train_test_split(htru_X, htru_Y, test_size=0.3)

# 十折交叉验证进行训练并计算错误率
kf = KFold(n_splits=10)
#定义列表
e_bag = [];bag_model = [];dicBag = [];Hoob = [];CorrGallary = []

#进行训练
for train, test in kf.split(np.arange(m)):
    X_train = htru_X[train]
    Y_train = htru_Y[train]
    X_test = htru_X[test]
    Y_test = htru_Y[test]
    err, corr_coefficient, testGallary, model,errknn, errtree, errgnb, errbp = Bagging(X_train, X_test, Y_train, Y_test)

    #计算集成学习的错误率
    for i in range(np.size(testGallary, axis=1)):
        dic_bag = Counter(np.array(testGallary)[:, i]).most_common(1)
        dicBag.append(dic_bag[0][0])
    count = 0
    n = len(Y_test)
    for i in range(n):
        if dicBag[i] == Y_test[i]:
            count +=1
    e_bag.append(1-count/n)
    bag_model.append(model)
    CorrGallary.append(corr_coefficient)
    Hoob.append(err)
    print("代码断点：K折交叉——每一折")
#集成学习器错误率
best = np.argmax(e_bag)
BagModel = bag_model[best]
errBag = e_bag[best]

#输出
print('WHN集合分类器——决策树错误率:')
print(errtree)
print('WHN集合分类器——KNN错误率:')
print(errknn)
print('WHN集合分类器——朴素贝叶斯错误率：')
print(errgnb)
print('WHN集合分类器——BP神经网络错误率:')
print(errbp)
print('WHN集合分类器——集成学习错误率：')
print(errBag)
print('系统多样性评价指标——相关系数：')
print(corr_coefficient)
