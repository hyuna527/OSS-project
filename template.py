#PLEASE WRITE THE GITHUB URL BELOW!
#

import sys
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score,accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier


def load_dataset(dataset_path):
	#To-Do: Implement this function
	return pd.read_csv(dataset_path)


def dataset_stat(dataset_df):	
	#To-Do: Implement this function
	#열개수 -1 이 특징점의 개수같은데, target열은 제외하는게 맞겠지?
	n_feats = dataset_df.shape[1] - 1
	n_class0 = len(dataset_df.loc[dataset_df['target'] == 0])
	n_class1 = len(dataset_df.loc[dataset_df['target'] == 1])
	return n_feats, n_class0, n_class1

def split_dataset(dataset_df, testset_size):
	#To-Do: Implement this function
	x = dataset_df.drop(columns="target",axis = 1)#라벨 제외
	y = dataset_df['target']#라벨 분류
	train_data, test_data, train_label, test_label = train_test_split(x,y,test_size = testset_size,random_state=1)#train/test 분류,일단 고정
	return train_data, test_data, train_label, test_label

def decision_tree_train_test(x_train, x_test, y_train, y_test):
	#To-Do: Implement this function
	dt_cls = DecisionTreeClassifier()
	dt_cls.fit(x_train,y_train)

	# 예측값 저장
	y_pred = dt_cls.predict(x_test)
	acc = accuracy_score(y_test,y_pred)
	prec = precision_score(y_test,y_pred)
	recall = recall_score(y_test,y_pred)
	return acc, prec, recall

def random_forest_train_test(x_train, x_test, y_train, y_test):
	#To-Do: Implement this function
	rf = RandomForestClassifier(n_estimators=100, oob_score=True, random_state=123456)
	rf.fit(x_train, y_train)

	#예측값 저장
	pred = rf.predict(x_test)
	acc = accuracy_score(y_test, pred)
	prec = precision_score(y_test,pred)
	recall = recall_score(y_test,pred)
	
	return acc, prec, recall

# def svm_train_test(x_train, x_test, y_train, y_test):
# 	#To-Do: Implement this function

def print_performances(acc, prec, recall):
	#Do not modify this function!
	print ("Accuracy: ", acc)
	print ("Precision: ", prec)
	print ("Recall: ", recall)

if __name__ == '__main__':
	#Do not modify the main script!
	data_path = sys.argv[1]
	data_df = load_dataset(data_path)

	n_feats, n_class0, n_class1 = dataset_stat(data_df)
	print ("Number of features: ", n_feats)
	print ("Number of class 0 data entries: ", n_class0)
	print ("Number of class 1 data entries: ", n_class1)

	print ("\nSplitting the dataset with the test size of ", float(sys.argv[2]))
	x_train, x_test, y_train, y_test = split_dataset(data_df, float(sys.argv[2]))

	acc, prec, recall = decision_tree_train_test(x_train, x_test, y_train, y_test)
	print ("\nDecision Tree Performances")
	print_performances(acc, prec, recall)

	acc, prec, recall = random_forest_train_test(x_train, x_test, y_train, y_test)
	print ("\nRandom Forest Performances")
	print_performances(acc, prec, recall)

	# acc, prec, recall = svm_train_test(x_train, x_test, y_train, y_test)
	# print ("\nSVM Performances")
	# print_performances(acc, prec, recall)