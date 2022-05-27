import numpy as np
import cvxpy

class SupportVectorMachine:
    def __init__(self, C, kernel, epsilon=1e-4):
        self.C = C
        self.epsilon = epsilon
        self.kernel = kernel

        # Hint: 你可以在训练后保存这些参数用于预测
        # SV即Support Vector，表示支持向量，SV_alpha为优化问题解出的alpha值，
        # SV_label表示支持向量样本的标签。
        # self.SV = None
        # self.SV_alpha = None
        # self.SV_label = None

    def KERNEL(self, x1, x2, d=2, sigma=1):
        #d for Poly, sigma for Gauss
        if self.kernel == 'Gauss':
            K = np.exp(-(np.sum((x1 - x2) ** 2)) / (2 * sigma ** 2))
        elif self.kernel == 'Linear':
            K = np.dot(x1,x2)
        elif self.kernel == 'Poly':
            K = (np.dot(x1,x2) + 1) ** d
        else:
            raise NotImplementedError()
        return K
    
    def fit(self, train_data, train_label):
        '''
        TODO：实现软间隔SVM训练算法
        train_data：训练数据，是(N, 7)的numpy二维数组，每一行为一个样本
        train_label：训练数据标签，是(N,)的numpy数组，和train_data按行对应
        '''

    def predict(self, test_data):
        '''
        TODO：实现软间隔SVM预测算法
        train_data：测试数据，是(M, 7)的numpy二维数组，每一行为一个样本
        必须返回一个(M,)的numpy数组，对应每个输入预测的标签，取值为1或-1表示正负例
        '''
        
        

# if __name__ == '__main__':
#     train_data = np.loadtxt('dataset/svm_train_data.csv', delimiter=',', skiprows=1)
#     train_label = np.loadtxt('dataset/svm_train_label.csv', delimiter=',', skiprows=1)
#     test_data = np.loadtxt('dataset/svm_test_data.csv', delimiter=',', skiprows=1)
#     test_label = np.loadtxt('dataset/svm_test_label.csv', delimiter=',', skiprows=1)

#     alg = SupportVectorMachine(1, 'Linear', 1e-4)

#     alg.fit(train_data=train_data, train_label=train_label)
#     pred = alg.predict(test_data).reshape(test_label.shape)

#     pred[pred >= 0] = 1
#     pred[pred < 0] = -1

#     print('acc: {}'.format(np.sum(pred == test_label) / test_data.shape[0]))