import numpy as np
import scipy.io as sio
import time
import timeit
import h5py
import cupy as cp

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
alpha = 0.25
beta = 0.5
a0 = (1 / alpha / dt / dt)
a1 = (beta /alpha / dt)
a2 = (1 / alpha / dt)
a3 = (1 / 2 / alpha -1)
a4 = (beta / alpha - 1)
a5 = (dt / 2 * (beta / alpha -2))
a6 = (dt * (1 - beta))
a7 = (dt * beta)
Ke = K + a0 * M + a1 * C
KeI = np.linalg.inv(Ke)
u = np.zeros((dofs , n))
v = np.zeros((dofs , n))
ac = np.zeros((dofs , n))
P = np.zeros((n,1))

t = cp.asarray(0)
M = cp.asarray(M)
C = cp.asarray(C)
K = cp.asarray(K)
diagM = cp.asarray(diagM)
dofs = cp.asarray(dofs)
dt = cp.asarray(dt)
P = cp.asarray(P)
u = cp.asarray(u)
v = cp.asarray(v)
ac = cp.asarray(ac)
a0 = cp.asarray(a0)
a1 = cp.asarray(a1)
a2 = cp.asarray(a2)
a3 = cp.asarray(a3)
a4 = cp.asarray(a4)
a5 = cp.asarray(a5)
a6 = cp.asarray(a6)
a7 = cp.asarray(a7)
KeI = cp.asarray(KeI)

print('start')
time_start = time.time()
for i in range(1 , n-1):
    t = 1 / 2000 * (i-1)
    P[i] = np.sin(2 * 3.1415926 * 2 * t)
    PP = -P[i] * diagM + np.dot(M , (a0 * u[:,i-1] + a2 * v[:,i-1] + a3 * ac[:,i-1])) + np.dot(C , (a1 * u[:,i-1] + a4 * v[:,i-1] + a5 * ac[:,i-1]))
    u[:, i] = np.dot(KeI, PP)
    ac[:, i] = a0 * (u[:, i] - u[:, i - 1]) - a3 * ac[:, i - 1] - a2 * v[:, i - 1]
    v[:, i]= v[:, i - 1] + a6 * ac[:, i - 1] + a7 * ac[:, i]
time_end = time.time()
print('计算结束,用时',time_end-time_start,'秒','平均每步用时',(time_end-time_start) / n,'秒')

ac = cp.asnumpy(ac)
dataNew = 'C://Users/BJUT/Desktop/TestSpeed/' + 'acpython.mat'
sio.savemat(dataNew, {'acpythonNMB':ac})
print('数值子结构部分保存完毕')