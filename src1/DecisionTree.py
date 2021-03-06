from random import sample
import numpy as np

ATTR_NUM           = 9
ATTR_VALUE_NUM_MIN = 2
ATTR_VALUE_NUM_MAX = 12
POSITIVE = 1
NEGATIVE = 0

class DecisionTree:
    def __init__(self): 
       self.root = TreeNode(None,None) # a dummy root

    def fit(self, train_features, train_labels):
        '''
        TODO: 实现决策树学习算法.
        train_features是维度为(训练样本数,属性数)的numpy数组
        train_labels是维度为(训练样本数, )的numpy数组
        '''
        # no attr has been used
        available_features = np.arange(0,ATTR_NUM)
        train_set = np.arange(0,train_labels.size)
        self.FitRecursive(train_set, train_features, train_labels, available_features, self.root)

    def predict(self, test_features):
        '''
        TODO: 实现决策树预测.
        test_features是维度为(测试样本数,属性数)的numpy数组
        该函数需要返回预测标签，返回一个维度为(测试样本数, )的numpy数组
        '''
        #print("Predicting test set...")
        sample_num = np.size(test_features,0)
        predict_labels = np.zeros(sample_num,dtype=int)
        #print("Test set size: ",sample_num)
        i = 0
        while i<sample_num:
            #print("Predicting ",i," of ",sample_num," test cases")
            result = self.PredictRecursive(test_features,i,self.root.children[0])
            predict_labels[i] = result
            i = i+1
        return predict_labels


    # PredictRecursive    
    def PredictRecursive(self, test_features, test_target, node):
        if node.label is not None:
            return node.label
        else:
            value = test_features[test_target][node.attr]
            #print("Checking ",node.attr,"=",value," attr for ",test_target)
            #print("Node list size: ",len(node.children))
            return self.PredictRecursive(test_features, test_target, node.children[value])

    # FitRecursive      recursive call for decision tree learning
    # @train_set        numpy array that includes training set indexes
    # @available_features    numpy array that indicates wether an attribute has been used for splitting 
    # @parent_node      the node that calls this round of recursion(so "parent" node)
    #                   node that a new best attr should be selected in this round and be added to parent node's children list
    def FitRecursive(self, train_set, train_features, train_labels, available_features, parent_node):
        # check if recursion should end
        # note that train_set will never be empty since later we will check before recursive call

        # all labels same, the label is the result(don't forget to link the node)
        same_label_flag = 1
        first_label = train_labels[train_set[0]]
        for i in train_set:
            if train_labels[i] != first_label:
                same_label_flag = 0
                break
        if same_label_flag == 1:
            node = TreeNode(None,first_label)
            parent_node.children.append(node)
            return
        
        no_attr_flag = 0
        same_attr_value_flag = 1
        pcount = 0
        ncount = 0
        # no attr for select OR all data has same value for available attrs, select label with more data as result(don't forget to link the node)
        for i in train_set:
                if train_labels[i] == 0:
                    ncount = ncount + 1
                else:
                    pcount = pcount + 1

        if available_features.size == 0:
            no_attr_flag = 1
        else:
            for i in train_set:
                for j in available_features:
                    if train_features[i][j] != train_features[0][j]:
                        same_attr_value_flag = 0
                        break
                if same_attr_value_flag == 0:
                    break    
        
        if no_attr_flag==1 or same_attr_value_flag==1 :
            if pcount >= ncount:
                node = TreeNode(None,POSITIVE)
            else: 
                node = TreeNode(None,NEGATIVE)
            parent_node.children.append(node)
            return 

        # select best attr
        best_attr,best_index = self.FindBestAttr(train_set,train_features, train_labels,available_features)

        
        # for each value of attr, create a leaf or recursive call
        attr_count_buckets = np.zeros(ATTR_VALUE_NUM_MAX,dtype=int)
        attr_pcount_buckets = np.zeros(ATTR_VALUE_NUM_MAX,dtype=int)
        attr_ncount_buckets = np.zeros(ATTR_VALUE_NUM_MAX,dtype=int)
        for i in train_set:
            attr_value = train_features[i][best_attr]
            label = train_labels[i]
            attr_count_buckets[attr_value] = attr_count_buckets[attr_value]+1
            if label == POSITIVE:
                attr_pcount_buckets[attr_value] = attr_pcount_buckets[attr_value]+1
            else:
                attr_ncount_buckets[attr_value] = attr_ncount_buckets[attr_value]+1    

        value = 0
        current_node = TreeNode(best_attr,None)
        parent_node.children.append(current_node)    
        while value <= ATTR_VALUE_NUM_MAX-1:
        # no data in the split, select label with more data as the result(don't forget to link the node)
            if attr_count_buckets[value] == 0: 
                if attr_pcount_buckets[value] >= attr_ncount_buckets[value]:
                    label = POSITIVE
                else:
                    label = NEGATIVE
                leaf_node = TreeNode(best_attr,label)
                current_node.children.append(leaf_node)

        # recursive call
            else:
                split_list = []
                for i in train_set:
                    if train_features[i][best_attr] == value:
                        split_list.append(i)
                split_set = np.array(split_list)
                self.FitRecursive(split_set, train_features, train_labels, np.delete(available_features,best_index), current_node)
                
            value = value + 1

        

    # Find the best splitting attribute for current dataset
    def FindBestAttr(self, train_set, train_features, train_labels, available_features):
        # for each available attr, calculate IG. Select the one with highest IG
        if available_features.size == 0:
            return None,None
        ig_max = -np.inf
        attr_id_best = 0
        attr_index_best = 0
        i = 0
        while i < available_features.size:
            attr_id = available_features[i]
            ig = self.IG(attr_id, train_set, train_features, train_labels)
            if ig > ig_max:
                ig_max  = ig
                attr_id_best = attr_id
                attr_index_best = i
            i = i + 1
        return attr_id_best,attr_index_best
    
    # Helpers for calculating information gain
    def log(self,base,x):
        return np.log(x)/np.log(base)

    def IG(self, attr_id, train_set, train_features, train_labels):
        # calculate entropy
        ent = self.I(attr_id, train_set, train_features, train_labels)
        # calculate conditional entropy
        cent = self.Remainder(attr_id, train_set, train_features, train_labels)
        ig = ent - cent
        return ig

    def I(self, attr_id, train_set, train_features, train_labels):
        #print(train_set)
        #print(attr_id)
        pcount = 0
        ncount = 0
        for i in train_set:
            if train_labels[i] == NEGATIVE:
                ncount = ncount + 1
            else: 
                pcount = pcount + 1
        pfreq = pcount/(pcount+ncount)
        nfreq = ncount/(pcount+ncount)
        if pcount == 0:
            result = - nfreq*self.log(2,nfreq)
        elif ncount==0:
            result = - pfreq*self.log(2,pfreq)
        else:
            result = -pfreq*self.log(2,pfreq) - nfreq*self.log(2,nfreq)
        return result

    def Remainder(self, attr_id, train_set, train_features, train_labels):
        train_buckets_pcount = np.zeros(ATTR_VALUE_NUM_MAX,dtype=int)
        train_buckets_ncount = np.zeros(ATTR_VALUE_NUM_MAX,dtype=int)
        # split into groups and calculate
        for i in train_set:
            attr_value = train_features[i][attr_id]
            if train_labels[i] == NEGATIVE:
                train_buckets_ncount[attr_value] = train_buckets_ncount[attr_value]+1
            else: 
                train_buckets_pcount[attr_value] = train_buckets_pcount[attr_value]+1
        j = 0
        result = 0
        while j<ATTR_VALUE_NUM_MAX:
            total_count = train_buckets_pcount[j]+train_buckets_ncount[j]   
            if total_count > 0:
                pfreq = train_buckets_pcount[j]/total_count
                nfreq = train_buckets_ncount[j]/total_count
                if pfreq==0:
                    result = result + (-nfreq*self.log(2,nfreq))*(total_count/train_set.size)
                elif nfreq==0:
                    result = result + (-pfreq*self.log(2,pfreq))*(total_count/train_set.size)
                else:
                    result = result + (-pfreq*self.log(2,pfreq) - nfreq*self.log(2,nfreq))*(total_count/train_set.size)
            j = j+1
        return result



# treenode: [attr, feat[attr] == 1, feat[attr] == 2, feat[attr] == 3]
class TreeNode:
    def __init__(self, attr, label):
        self.children=[]
        self.attr = attr
        self.label = label