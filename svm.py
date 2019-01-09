# SVM for educational purposes
# If you are using ipython I would suggest to use different segments for each comment section.
# You can play with the Gaussian data by changing the mean and the variance and observe how the decision boundary changes with respect to that.
# While you change the input data you may also adjust your Kernel type and see how it effects the decision boundary.


import numpy as np
import random, math
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import scipy.linalg

### Let us create our labeled training data from a Gaussian distribution ###

classA = np.concatenate((np.random.randn(10,2) * 0.9 + [1.5,0.5], np.random.randn(10,2)* 0.2 + [-1.5,0.5]))
classB = np.random.randn(20, 2) * 0.3 + [0.0, -0.5]

inputs = np.concatenate((classA, classB))
targets = np.concatenate((np.ones(classA.shape[0]),-np.ones(classB.shape[0])))

N = inputs.shape[0]  #number of rows (samples)

permute = list(range(N))
random.shuffle(permute)
inputs = inputs[permute,:]
targets = targets [permute]

x = inputs


plt.plot ([p[0] for p in classA] , [p[1] for p in classA] , 'b.')
plt.plot ([p[0] for p in classB] , [p[1] for p in classB] , 'r.')

plt.axis('equal')   #force same scale on both axis
plt.savefig('svmplot1.pdf')
plt.show()


####  Let us define different Kernels for different datasets that we are going to use ###


def linear_kernel(x1, x2):
    return np.dot(x1, x2)


def polynomial_kernel(x, y, p=7):
    return (1 + np.dot(x, y)) ** p

def gaussian_kernel(x, y, sigma=0.9):
    return np.exp(-np.linalg.norm(x-y)**2 / (2 * (sigma ** 2)))
    
    
#######################################################################################


P = np.zeros((N,N))

for i in range(N):
    for j in range(N):
        P[i,j] = targets[i] * targets [j] * polynomial_kernel(x[i], x[j])
        
 #####################################################################################
 
 def zerofun(alpha):
    return(np.dot(alpha, targets))
    
 

start = np.zeros(N)
B=[(0, None) for b in range(N)]
cons = {'type':'eq', 'fun':zerofun}

#######################################################################################

def objective(alpha):
    result = 0
    sum_alpha = np.sum(alpha)
    
    for i in range(N):
        for m in range(N):
            result += alpha[i] * alpha[m] * P[i,m]
    result = 0.5 * result - sum_alpha
    return(result)
#######################################################################################
### Let us find our non zero vectors which they will be the support vectors ###

ret = minimize(objective, start, bounds = B, constraints = cons)
alphas = ret['x']
alphas


######################################################################################


j = 0
non_alpha = np.zeros(N)
non_inputs = np.zeros((N,2))
non_targets = np.zeros(N)




for i in range(N):
    if alphas[i] >= 10e-5:
        non_alpha[j] = alphas[i]
        non_inputs[j,0] = inputs[i,0]
        non_inputs[j,1] = inputs[i,1]
        non_targets[j] = targets[i]
        j += 1
    
non_inputs       
non_alpha_cut = np.zeros(j)
non_targets_cut = np.zeros(j)
non_inputs_cut = np.zeros((j, 2))




non_alpha_cut = non_alpha[0:j]
non_targets_cut = non_targets[0:j]
non_inputs_cut[:,0] = non_inputs[0:j,0]
non_inputs_cut[:,1] = non_inputs[0:j,1]



alpha_ind_max = np.argmax(non_alpha_cut)
inp_max = non_inputs_cut[alpha_ind_max]
inp_max_target = non_targets_cut[alpha_ind_max]


print (non_alpha_cut)
print  ('.')
print (non_inputs_cut)
print  ('.')
print  (inp_max)
print  ('.')
print  (non_targets_cut)
print  ('.')
print  (inp_max_target)

#############################################################################


b=0
for i in range(j):
    b += non_alpha_cut[i] * non_targets_cut[i] * polynomial_kernel(inp_max, non_inputs_cut[i]) 
    
    
print(b)
print('.')    
b = b - inp_max_target   
b,j, non_targets_cut

############################################################################

def indicator2(x, y):
    iden2 = 0
    s2 = np.zeros((1,2))
    s2[0,0] = x
    s2[0,1] = y
    #print(s2)
    for i in range(j):
        iden2 += non_alpha_cut[i] * non_targets_cut[i] * polynomial_kernel(s2, non_inputs_cut[i]) 
    iden2 = iden2 - b
    #print(iden2)
    if iden2 > 10e-1:
        return 1
    elif iden2 < -10e-1:
        return -1
    else:
        return 0
        
 #############################################################################
 
 
xgrid = np.linspace(-4,4, 300)
ygrid = np.linspace(-3,3, 300)

grid = np.array([[indicator2(x,y)
                  for x in xgrid]
                 for y in ygrid])

               
fig = plt.figure()
#plt.contour(xgrid, ygrid, grid, (-1, 0, 1), colors = ('red' , 'black', 'blue'), linewidths = (1,3,1))
plt.contour(xgrid, ygrid, grid, colors = ('red' , 'black', 'blue'))
plt.plot ([p[0] for p in classA] , [p[1] for p in classA] , 'b.')
plt.plot ([p[0] for p in classB] , [p[1] for p in classB] , 'r.')
plt.plot ([p[0] for p in non_inputs_cut] , [p[1] for p in non_inputs_cut] , 'y+')
#plt.plot (inp_max[0,0] , inp_max[0,1] , 'yo')
#plt.scatter(xgrid, ygrid)
plt.grid()
fig.suptitle('SVM Polynomial Kernel p_7_0.9_0.2_0.3', fontsize=16)
plt.savefig('SVM Polynomial Kernel p_7_0.9_0.2_0.3.svg')
plt.show()
