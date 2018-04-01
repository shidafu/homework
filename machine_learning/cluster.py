'''
Cluster method
author: Leon, date: 2017.12.04
'''
'''
Data typy:
    m×n 2D array means m vectors in a bach, each vector has n elements
   [[x11,x12...x1n], -> vector 1
    [x21,x22...x2n], -> vector 2
         :                :
    [x1m,x1m...x1m]] -> vector m
Element operation:
    e_func(m×n x, float y)= m×n array,  such as e_max(x,y)[i,j]= max(x[i,j],y)
    e_func(m×n x, m×n y)= m×n array,  such as e_max(x,y)[i,j]= max(x[i,j],y[i,j])
Vector operation:
    v_func(m×n x)= m×? array,  such as v_l2norm(x,y)[i,1]= l2norm(x[i,:])
    v_func(m×n x, k×n y)= m×k array,  such as v_inner(x,y)= x·y'
Batch operation:
    b_func(m×n x)= 1×n array,  such as b_norm(x)[1,i]= norm(x[:,i])

Data typy in model:
Without bias:inputs·weights'=outputs
   Data:           inputs                weights              outputs
   Format:    Batchs×Inputs(B-I)   Outputs×Inputs(O-I)   Batchs×Outputs(B-O)
                 [[2,1,0],             [[1.0,0.0,0.0],       [[2.0,1.0],
   Demo:          [1,2,1],              [0.0,1.0,0.0]]        [1.0,2.0],
                     :                                           :
                  [3,0,3]]                                    [3.0,0.0]]
Or include bias:input_ex·weights_ex'=outputs_ex
   Data:           inputs                weights                     outputs
   Format:  Batchs×(Inputs+1)(B-I1) Outputs×(Inputs+1)(O-I1)  Batchs×Outputs(B-O)
              [[2,1,0,const=1],         [1.0,0.0,0.0,b0],        [[2.0+b0,1.0+b1],
   Demo:       [1,2,1,const=1],         [0.0,1.0,0.0,b1]]         [1.0+b0,2.0+b1],
                     :                                                  :
               [3,0,3,const=1]]                                   [3.0+b0,0.0+b1]]
'''

import numpy as np

'''
Vector Operations
'''
def v_euclid_dist(x, y):
    '''
    Euclidean distance of vectors,
    '''
    x_y_mat = np.zeros((len(x),len(y)))
    for j in range(len(x)):
        for i in range(len(y)):
            x_y_mat[j,i] = np.linalg.norm(x[j] - y[i], ord=2, axis=0, keepdims=True)
    return x_y_mat


def v_sq_euclid_dist(x,y):
    '''
    Squared Euclidean distance of vectors,
    '''
    x_y_mat = np.zeros((len(x),len(y)))
    for j in range(len(x)):
        for i in range(len(y)):
            x_y_mat[j,i] = np.linalg.norm(x[j] - y[i], ord=2, axis=0, keepdims=True)
            x_y_mat[j,i] = x_y_mat[j,i]*x_y_mat[j,i]
    return x_y_mat


def v_manhattan_dist(x, y):
    '''
    Manhattan distance of vectors,
    '''
    x_y_mat = np.zeros((len(x),len(y)))
    for j in range(len(x)):
        for i in range(len(y)):
            x_y_mat[j,i] = np.linalg.norm(x[j] - y[i], ord=1, axis=0, keepdims=True)
    return x_y_mat


def v_maximum_dist(x,y):
    '''
    Maximum distance of vectors,
    '''
    x_y_mat = np.zeros((len(x),len(y)))
    for j in range(len(x)):
        for i in range(len(y)):
            x_y_mat[j,i] = np.linalg.norm(x[j] - y[i], ord=np.inf, axis=0, keepdims=True)
    return x_y_mat


def v_inner(x,y):
    '''
    Inner product distance of vectors,
    '''
    return np.dot(x,np.transpose(y))


def v_cosin(x,y):
    '''
    Cosin distance of vectors,
    '''
    x_norm = np.linalg.norm(x, ord=2, axis=1, keepdims=True)
    y_norm = np.linalg.norm(y, ord=2, axis=1, keepdims=True)
    x = np.divide(x,x_norm)
    y = np.divide(y,y_norm)
    return np.dot(x,np.transpose(y))


def v_l2norm(x):
    '''
    L2 normalize of vectors,
    '''
    x_l2norm = np.linalg.norm(x, ord=2, axis=1, keepdims=True)
    return np.divide(x,x_l2norm)

'''
Element Operations
'''
def e_liner(x):
    return x


def e_relu(x):
    return np.maximum(x,0)


def e_compete(x,ismax = True):
    x_c = np.zeros_like(x)
    if ismax == False:
        x = -1*x
    for j,xx in enumerate(x):
        x_c[j,np.argmax(xx)] = 1.0
    return x_c


def e_sigmoid(x):
    return np.reciprocal(np.add(np.exp(np.negative(x)),1))


def e_tanh(x):
    return np.tanh(x)


def e_softplus(x):
    return np.log(np.add(1, np.exp(x)))


def e_gauss(x,a=[0,1]):
    return np.exp(np.negative(0.5*np.square((x-a[0])/a[1])))


def v_softmax(x):
    x_y_mat = np.zeros((len(x),len(y)))
    for j in range(len(x)):
        x_soft = np.exp(x[j])
        x[j] = np.divide(x_soft,np.sum(x_soft))
    return x


class k_means:
    '''
    k_means algorithm,
    '''
    def __init__(self, a_list, dist_funx = v_euclid_dist):
        #self.x_array = np.array(x_list)
        self.a_array = np.array(a_list)
        self.m = len(self.a_array[0])
        self.k = len(self.a_array)
        self.dist_funx = dist_funx
        print("k_means __init__("+ str(self.k) +"-means)")

    def predict(self, x_list):
        c_list=[]
        for x in x_list:
            c_x = np.argmin(self.dist_funx([x], self.a_array))
            c_list.append(c_x)
        return np.array(c_list)

    def update(self, x_list,  c_list = None):
        if c_list is None:
            c_list=[]
            for x in x_list:
                print(self.dist_funx([x], self.a_array))
                c_x = np.argmin(self.dist_funx([x], self.a_array))
                c_list.append(c_x)
        a_array = self.a_array * 0
        a_num = np.zeros(self.k)
        for i,c in enumerate(c_list):
            a_array[c] += x_list[i]
            a_num[c] += 1.0
        for i,a in enumerate(a_array):
            a_array[i] /= a_num[i]
        up_dist = np.linalg.norm(self.a_array - a_array)
        isend = (up_dist<=0.0000000000001)
        self.a_array = a_array
        return c_list,a_array,isend,up_dist

# 作业：
# class k_means_plus
# class k-mediod

class GMM:
    '''
    Gaussian mixture models algorithm,
    '''
    def __init__(self, x_list, a_list):
        self.x_array = np.array(x_list)
        self.n = len(self.x_array)
        self.m = len(self.x_array[0])
        self.para_sd_m = np.power(2*np.pi,-0.5*self.m)
        self.a_array = np.array(a_list)
        self.k = len(self.a_array)
        self.c_mat = np.random.random_sample((self.n,self.k))+0.5
        for j in range(self.k):
            self.c_mat[j,:] = self.c_mat[j,:]/(np.sum(self.c_mat[j,:]))
        self.update(self.x_array)
        print("GMM __init__("+ str(self.k) +"-components)")

    def v_gauss_possib(self,x,index):
        # Gaussin distribute of input,
        x_a = x-self.a_array[index]
        p_x = np.dot(np.dot(x_a, np.linalg.inv(self.sd_array[index])), np.transpose(x_a))
        p_x = self.para_sd_m * self.sdet_array[index]*np.exp(-0.5*p_x)
        return p_x

    def predict(self, x_list):
        # E-step
        self.x_array = np.array(x_list)
        self.c_mat=[]
        for i in range(len(self.x_array)):
            c_i=[]
            c_sum=0
            for j in range(self.k):
                c_i_j = self.pi_array[j] * self.v_gauss_possib(self.x_array[i],j)
                c_i.append(c_i_j)
                c_sum+=c_i_j
            self.c_mat.append(c_i/c_sum)
        self.c_mat = np.array(self.c_mat)
        c_array = np.argmax(self.c_mat, 1)
        return c_array

    def update(self, x_list):
        # M-step
        x_array = np.array(x_list)
        #x_array = np.array(x_list)
        # Nk
        N_array = np.sum(self.c_mat, axis=0)
        # uk
        a_array = np.dot(np.transpose(self.c_mat),x_array)
        for j in range(self.k):
            a_array[j,:] = a_array[j,:]/N_array[j]
        # 收敛
        up_dist = np.linalg.norm(self.a_array - a_array)
        isend = (up_dist<=0.0000000000001)
        # update
        self.a_array = a_array
        # sd
        self.sd_array = []
        self.sdet_array = []
        for j in range(self.k):
            sd = np.zeros((self.m,self.m))
            for i in range(len(x_array)):
                x_a=x_array[i]-self.a_array[j]
                sd = sd + self.c_mat[i,j]* np.dot(np.transpose([x_a]),[x_a])
            sd = sd/N_array[j]
            self.sd_array.append(sd)
            self.sdet_array.append(np.power(np.linalg.det(sd),-0.5))
        self.sd_array = np.array(self.sd_array)
        self.sdet_array = np.array(self.sdet_array)
        # pik
        self.pi_array = N_array/ float(len(x_array))
        # E-step
        c_array = self.predict(x_array)
        return c_array,a_array,isend,up_dist




