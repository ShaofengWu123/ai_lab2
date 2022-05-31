from utils import *
from DecisionTree import DecisionTree
from SVM import SupportVectorMachine


def test_decisiontree():
    train_features, train_labels, test_features, test_labels = load_decisiontree_dataset()
    model = DecisionTree()
    model.fit(train_features, train_labels)
    results = model.predict(test_features)
    # results = np.random.randint(2, size=56)
    print('DecisionTree acc: {:.2f}%'.format(get_acc(results, test_labels) * 100))

def test_svm(C=1, kernel='Linear', epsilon = 1e-4):
    train_features, train_labels, test_features, test_labels = load_svm_dataset()
    model = SupportVectorMachine(C, kernel, epsilon)
    model.fit(train_features, train_labels)
    pred = model.predict(test_features)
    print('SVM acc: {:.2f}%'.format(get_acc(pred, test_labels.reshape(-1,)) * 100))

def debug_decisiontree(train_set,train_features,train_labels):
    #train_features, train_labels, test_features, test_labels = load_decisiontree_dataset()
    print(train_set)
    print(train_features)
    print(train_labels)
    model = DecisionTree()
    print(model.IG(8, train_set, train_features, train_labels))
    #available_features = np.arange(0,9)
    available_features = np.array([0,2,8,6,7,3,4,5,1])
    print(model.FindBestAttr(train_set, train_features, train_labels, available_features))
    model.fit(train_features, train_labels)
    print(model.predict(train_features))
    

if __name__=='__main__':
    test_decisiontree() 
    # test_svm(1, 'Linear', 1e-4)

    # from slides
    # debug_train_features = np.array([[1,0,0,1,1,2,0,1,0],
    #                                  [1,0,0,1,2,0,0,0,1],
    #                                  [0,1,0,0,1,0,0,0,2],
    #                                  [1,0,1,1,2,0,0,0,1],
    #                                  [1,0,1,0,2,2,0,1,0],
    #                                  [0,1,0,1,1,1,1,1,3],
    #                                  [0,1,0,0,0,0,1,0,2],
    #                                  [0,0,0,1,1,1,1,1,1],
    #                                  [0,1,1,0,2,0,1,0,2],
    #                                  [1,1,1,1,2,2,0,1,3],
    #                                  [0,0,0,0,0,0,0,0,1],
    #                                  [1,1,1,1,2,0,0,0,2]
    #                                  ]
    #                                  )
    # debug_train_labels = np.array([1,0,1,1,0,1,0,1,0,0,0,1])
    # debug_train_set = np.arange(0,12)
    # debug_decisiontree(debug_train_set,debug_train_features,debug_train_labels)
    
