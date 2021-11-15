import numpy as np
import scipy.io as sio
import time
import timeit
import h5py
print('GPU中心差分法，开始')
MCK = sio.loadmat('MCK2592.mat')
# MCK = h5py.File('MCK2592.mat')
M = np.asarray(MCK['M'])
C = np.asarray(MCK['C'])
K = np.asarray(MCK['K'])
print('数据读取完毕')
dofs = 2592
M = M[0 : (dofs),0 : (dofs)]
K = K[0 : (dofs),0 : (dofs)]
C = C[0 : (dofs),0 : (dofs)]
dofs = M.shape[0]
diagM = M[np.diag_indices(dofs)].T
dt = 0.001
n = 2000
Ke=M/(dt*dt)+((C)/(2*dt))
a = K - (2 * M) / (dt*dt)
b=M/(dt*dt) - C/(2*dt)
u = np.zeros((dofs , n))
v = np.zeros((dofs , n))
ac = np.zeros((dofs , n))
P = np.zeros((n,1))
KeI = np.linalg.inv(Ke)
time_start = time.time()
print('start')
for i in range(1 , n-1):
    t = 1/2000*(i-1)
    P[i] = np.sin(2*3.1415926*2*t)
    PP = -P[i] * diagM - np.dot(a , u[: , i]) - np.dot(b , u[: , i-1])
    u[:,i+1] = np.dot(KeI, PP)
    v[: , i] = (u[: , i+1] - u[: , i-1]) / (dt*2)
    ac[: , i] = (u[: , i+1] - 2 * u[: , i] + u[: , i-1]) / (dt*dt)
time_end = time.time()
print('计算结束,用时',time_end-time_start,'秒','平均每步用时',(time_end-time_start) / n,'秒')

dataNew = 'C://Users/BJUT/Desktop/TestSpeed/' + 'acpython.mat'
sio.savemat(dataNew, {'acpythonCDM':ac})
print('数值子结构部分保存完毕')