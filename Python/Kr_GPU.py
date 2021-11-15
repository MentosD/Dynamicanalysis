import numpy as np
import scipy.io as sio
import time
import timeit
import h5py
import cupy as cp
print('GPU中心差分法，GPU开始')
# MCK = sio.loadmat('MCK2592.mat')
MCK = h5py.File('MCK24375.mat')
M = np.asarray(MCK['M'])
C = np.asarray(MCK['C'])
K = np.asarray(MCK['K'])
print('数据读取完毕')
dofs = 24375
M = M[0 : (dofs),0 : (dofs)]
K = K[0 : (dofs),0 : (dofs)]
C = C[0 : (dofs),0 : (dofs)]
dofs = M.shape[0]
diagM = M[np.diag_indices(dofs)].T
dt = 0.001
n = 200
gama = 7/6
beta = 25/36
af = 1 / 6
am = -0.5
t = 0
u = np.zeros((dofs , n))
v = np.zeros((dofs , n))
ac = np.zeros((dofs , n))
P = np.zeros((n,1))
MMI = np.linalg.inv(M * (1 - am) + C * gama * dt * (1 - af) + K * beta * dt * dt * (1 - af))
A1 = (M * am + C * (1 - af) * dt * (1 - gama) + K * (1 - af) * dt * dt * (0.5 - beta))
A2 = (C * (1 - af) + C * af + K * (1 - af) * dt)
A3 =(K * (1 - af) + K * af)

t = cp.asarray(t)
M = cp.asarray(M)
C = cp.asarray(C)
K = cp.asarray(K)
diagM = cp.asarray(diagM)
dofs = cp.asarray(dofs)
dt = cp.asarray(dt)
gama = cp.asarray(gama)
beta = cp.asarray(beta)
af = cp.asarray(af)
am = cp.asarray(am)
u = cp.asarray(u)
v = cp.asarray(v)
ac = cp.asarray(ac)
P = cp.asarray(P)
MMI = cp.asarray(MMI)
A1 = cp.asarray(A1)
A2 = cp.asarray(A2)
A3 = cp.asarray(A3)

print('start')
time_start = time.time()
for i in range(1 , n-1):
    t = 1/2000*(i-1)
    P[i] = cp.sin(2*3.1415926*2*t)
    PP = -((1 - af) * P[i] + af * P[i-1]) * diagM - cp.dot(ac[: , i-1] , A1) - cp.dot(v[: , i-1] , A2) - cp.dot(u[: , i-1] , A3)
    ac[: , i] = cp.dot(MMI , PP)
    v[: , i] = v[: , i-1] + dt * ((1 - gama) * ac[: , i-1] + gama * ac[: , i])
    u[: , i] = u[: , i-1] + dt * v[: , i-1] + (dt * dt) * ((0.5 - beta) * ac[: , i-1] + beta * ac[: , i])
time_end = time.time()
print('计算结束,用时',time_end-time_start,'秒','平均每步用时',(time_end-time_start) / n,'秒')

ac = cp.asnumpy(ac)
dataNew = 'C://Users/BJUT/Desktop/TestSpeed/' + 'acpython.mat'
sio.savemat(dataNew, {'acpythonKRGPU':ac})
print('数值子结构部分保存完毕')
