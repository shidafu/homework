'''
Vector quantization method
author: Leon, date: 2017.12.09
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
    # Euclidean distance of vectors,
    x_y_mat = np.zeros((len(x),len(y)))
    for j in range(len(x)):
        for i in range(len(y)):
            x_y_mat[j,i] = np.linalg.norm(x[j] - y[i], ord=2, axis=0, keepdims=True)
    return x_y_mat


def v_sq_euclid_dist(x,y):
    # Squared Euclidean distance of vectors,
    x_y_mat = np.zeros((len(x),len(y)))
    for j in range(len(x)):
        for i in range(len(y)):
            x_y_mat[j,i] = np.linalg.norm(x[j] - y[i], ord=2, axis=0, keepdims=True)
            x_y_mat[j,i] = x_y_mat[j,i]*x_y_mat[j,i]
    return x_y_mat


def v_manhattan_dist(x, y):
    # Manhattan distance of vectors,
    x_y_mat = np.zeros((len(x),len(y)))
    for j in range(len(x)):
        for i in range(len(y)):
            x_y_mat[j,i] = np.linalg.norm(x[j] - y[i], ord=1, axis=0, keepdims=True)
    return x_y_mat


def v_maximum_dist(x,y):
    # Maximum distance of vectors,
    x_y_mat = np.zeros((len(x),len(y)))
    for j in range(len(x)):
        for i in range(len(y)):
            x_y_mat[j,i] = np.linalg.norm(x[j] - y[i], ord=np.inf, axis=0, keepdims=True)
    return x_y_mat


def v_inner(x,y):
    # Inner product distance of vectors,
    return np.dot(x,np.transpose(y))


def v_cosin(x,y):
    # Cosin distance of vectors,
    x_norm = np.linalg.norm(x, ord=2, axis=1, keepdims=True)
    y_norm = np.linalg.norm(y, ord=2, axis=1, keepdims=True)
    x = np.divide(x,x_norm)
    y = np.divide(y,y_norm)
    return np.dot(x,np.transpose(y))


def v_l2norm(x):
    # L2 normalize of vectors,
    x_l2norm = np.linalg.norm(x, ord=2, axis=1, keepdims=True)
    return np.divide(x,x_l2norm)



# Grid function
def grid_2d(shape):
    # Make 2d grid,
    height=shape[0]
    width=shape[1]
    yx=[]
    for h in range(height):
        for w in range(width):
            yx.append([h,w])
    yx = np.array(yx)
    return yx

# Radius function
def radius_mexhat(x, h_sd=[1.0, 1.0, -0.1, 3.0]):
    up = h_sd[0] * np.exp(np.negative(0.5 * np.square(x / h_sd[1]))) / (h_sd[1] * np.sqrt(2 * np.pi))
    down = h_sd[2] * np.exp(np.negative(0.5 * np.square(x / h_sd[3]))) / (h_sd[3] * np.sqrt(2 * np.pi))
    y = up - down
    return y

def radius_gausshat(x, h_sd=[1.0, 1.0, 0.0]):
    y = h_sd[0] * np.exp(np.negative(0.5 * np.square(x / h_sd[1]))) / (h_sd[1] * np.sqrt(2 * np.pi))
    if h_sd[2] > 0.0:
        y -= np.mean(y)
    return y

def radius_tophat(x, h_sd=[1.0, 1.0, 3.0]):
    up_val = h_sd[0] * 0.5 / h_sd[1]
    down_val = -h_sd[0] * 0.5 / h_sd[2]
    up = np.zeros_like(x, np.float)
    down = np.zeros_like(x, np.float)
    up[np.where(np.abs(x) < (h_sd[1]))] = up_val
    down[np.where(np.abs(x) < (h_sd[2]))] = down_val
    y = up + down
    return y

def radius_chefhat(x, h_sd=[1.0, 1.0, 1.0]):
    up_val = h_sd[0] * 0.5 / h_sd[1]
    up = np.zeros_like(x, np.float)
    up[np.where(np.abs(x) < (h_sd[1]))] = up_val
    y = up
    if h_sd[2] > 0.0:
        y -= np.mean(y)
    return y

# Distant function
def dist_euclid(x, y):
    # Euclidean distance of vectors,
    x_y_mat = np.zeros((len(x), len(y)))
    for j in range(len(x)):
        for i in range(len(y)):
            x_y_mat[j, i] = np.linalg.norm(x[j] - y[i], ord=2, axis=0, keepdims=True)
    return x_y_mat

# Vector functiont
def vec_euclid(x, y):
    # Euclidean distance of vectors,
    x_y_mat = np.zeros((len(x),len(y)))
    for j in range(len(x)):
        for i in range(len(y)):
            x_y_mat[j,i] = np.linalg.norm(x[j] - y[i], ord=2, axis=0, keepdims=True)
    return x_y_mat

def vec_inner(x,y, h_sd=[1.0, 1.0]):
    # Inner product distance of vectors,
    return h_sd[0] *np.dot(x,np.transpose(y))

# Active functiont
def act_liner(x, h_sd=[1.0, 1.0]):
    return x

def act_relu(x, h_sd=[1.0, 1.0]):
    return np.maximum(x, 0)

def act_compete(x, h_sd=[1.0, 1.0]):
    x_c = np.zeros_like(x)
    for j, xx in enumerate(x):
        x_c[j, np.argmax(xx)] = 1.0
    return x_c

def act_sigmoid(x, h_sd=[1.0, 1.0]):
    return np.reciprocal(np.add(np.exp(np.negative(x)), 1))

def act_tanh(x, h_sd=[1.0, 1.0]):
    return np.tanh(x)

def act_softplus(x, h_sd=[1.0, 1.0]):
    return np.log(np.add(1, np.exp(x)))

def act_gauss(x, h_sd=[1.0, 0.0, 1.0]):
    return h_sd[0]*np.exp(np.negative(0.5 * np.square((x - h_sd[1]) / h_sd[2])))


class sofm:
    # sofm network,
    def __init__(self, n_in,
                 grid_func=grid_2d, grid_para=[8,8],
                 dist_func=dist_euclid,
                 radius_func=radius_mexhat, radius_para=[1.0, 1.0, 3.0],
                 vec_func=vec_euclid,
                 act_func=act_gauss, act_para=[1.0, 0.0, 1.0]):
        self.n_in = n_in
        self.xy = grid_func(grid_para)
        self.n_out = len(self.xy)
        mu, sigma = 0, 1.0
        self.w = np.random.normal(mu, sigma, [self.n_out,self.n_in])
        self.dist_func = dist_func
        self.radius_func = radius_func
        self.dist = np.zeros((self.n_out,self.n_out))
        self.radius = np.zeros((self.n_out,self.n_out))
        for i in range(self.n_out):
            for j in range(self.n_out):
                self.dist[i, j]= self.dist_func([self.xy[i]],[self.xy[j]])[0,0]
                self.radius[i, j]= self.radius_func(self.dist[i, j], radius_para)
        self.vec_func = vec_func
        self.act_func = act_func
        self.act_para = act_para
        print("sofm __init__()")

    def predict(self, x_list):
        x_list = np.array(x_list)
        y_list = self.act_func(self.vec_func(x_list,self.w),self.act_para)
        maxid_list = np.argmax(y_list, axis=1)
        return y_list,maxid_list

    def update(self, x_list, ratio=1.0):
        x_list = np.array(x_list)
        batch=len(x_list)
        y_list,maxid_list = self.predict(x_list)
        # print(y_list)
        # print(maxid_list)
        for o in range(self.n_out):
            # print('w[o, :]',end='')
            # print(self.w[o,:])
            wo=np.zeros_like(self.w[o,:])
            # print('wo',end='')
            # print(wo)
            for b, m in enumerate(maxid_list):
                wo+=(1-ratio*self.radius[o, m])*self.w[o,:]+ratio*self.radius[o, m]*x_list[b,:]
                # print('wo',end='')
                # print(self.radius[o, m],end='')
                # print(wo)
            # print(wo)
            # print('w[o, :]',end='')
            # print(self.w[o,:])
            # print('wo/batch',end='')
            # print(wo/batch)
            self.w[o,:]= wo/batch
            # print('w[o, :]2',end='')
            # print(self.w[o,:])
        return self.w

