import numpy as np
import cvxpy as cvx

class SupportVectorMachine:
    def __init__(self, C, kernel, epsilon=1e-4):
        self.C = C # trade-off parameter
        self.epsilon = epsilon
        self.kernel = kernel

        # Hint: 你可以在训练后保存这些参数用于预测
        # SV即Support Vector，表示支持向量，SV_alpha为优化问题解出的alpha值，
        # SV_label表示支持向量样本的标签。
        self.SV = []
        self.SV_alpha = []
        self.SV_label = []

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
    
    def kernel_func(self, x , d=2, sigma=1):
        gram  = np.zeros(len(x)*len(x),dtype=float).reshape(len(x),len(x))
        for i in range(len(x)):
            for j in range(len(x)):
                gram[i][j] = self.KERNEL(x[i],x[j])
        return gram

    def fit(self, train_data, train_label):
        '''
        TODO：实现软间隔SVM训练算法
        train_data：训练数据，是(N, 7)的numpy二维数组，每一行为一个样本
        train_label：训练数据标签，是(N,)的numpy数组，和train_data按行对应
        '''
        m = len(train_label) # # of samples
        d = train_data.shape[1] # dim of samples
        x = train_data # training samples
        y = []
        for i in range(m):
            y.append([train_label[i]])
        y = np.array(y)
        alpha = cvx.Variable(shape=(m,1),pos=True) # lagrange multiplier
        #G = np.matmul(y*x, (y*x).T) # Gram matrix
        G = self.kernel_func(x)
        objective = cvx.Maximize(cvx.sum(alpha)-(1/2)*cvx.quad_form(alpha, y@y.T*G))
        constraints = [alpha <= self.C, cvx.sum(cvx.multiply(alpha,y)) == 0] # box constraint
        prob = cvx.Problem(objective, constraints)
        result = prob.solve(verbose=False,solver=cvx.ECOS)
        
        i = 0
        a = alpha.value
        while i < a.size:
            if a[i][0] < self.epsilon:
                a[i][0] = 0
            else:
                self.SV.append(train_data[i])
                self.SV_alpha.append(a[i][0])
                self.SV_label.append(train_label[i])
            i = i + 1
        # print(self.SV)
        # print(self.SV_alpha)
        # print(self.SV_label)



    def predict(self, test_data):
        '''
        TODO：实现软间隔SVM预测算法
        train_data：测试数据，是(M, 7)的numpy二维数组，每一行为一个样本
        必须返回一个(M,)的numpy数组，对应每个输入预测的标签，取值为1或-1表示正负例
        '''
        predict_labels = []
        w = np.zeros(7,dtype = float)
        b = 0

        for i in range(len(self.SV)):
            w = w + self.SV_alpha[i] * self.SV_label[i] * self.SV[i]
        b = self.SV_label[0] - np.inner(w,self.SV[0])
        for i in test_data:
            predict_value = np.inner(w,i) + b
            if predict_value<0:
                predict_labels.append(-1)
            else:
                predict_labels.append(1)

        return predict_labels

        

# if __name__ == '__main__':
#     train_data = np.loadtxt('dataset/svm/svm_train_data.csv', delimiter=',', skiprows=1)
#     train_label = np.loadtxt('dataset/svm/svm_train_label.csv', delimiter=',', skiprows=1)
#     test_data = np.loadtxt('dataset/svm/svm_test_data.csv', delimiter=',', skiprows=1)
#     test_label = np.loadtxt('dataset/svm/svm_test_label.csv', delimiter=',', skiprows=1)

#     alg = SupportVectorMachine(1, 'Linear', 1e-4)

#     alg.fit(train_data=train_data, train_label=train_label)
#     pred = alg.predict(test_data)

#     print(pred)
#     print(test_label)
#     print('acc: {}'.format(np.sum(pred == test_label) / test_data.shape[0]))