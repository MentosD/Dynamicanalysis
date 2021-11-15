clear
clc
load('MCK5400.mat');                                        % import MCK data

dofs = 1                                                             % set DOFs
M = M(1:dofs,1:dofs);                                       % cut DOFs
C = C(1:dofs,1:dofs);
K = K(1:dofs,1:dofs);

dofs = length(M);
diagM = diag(1:dofs);
dt = 0.001;
Ke=M/(dt^2)+((C)/(2*dt));                       
a = K - (2 * M) / (dt)^2;
b=M/dt^2 - C/(2*dt);
u = zeros(dofs , 2000);
v = zeros(dofs , 2000);
ac = zeros(dofs , 2000);

for i = 2 : 2000
    t = 1/2000*(i-2);
    P(i) = sin (2*pi*2*t);
    PP = -P(i) * diagM - a * u(: , i) - b * u(: , i-1);
    u(:,i+1)=Ke \ PP;                            
    v(: , i) = (u(: , i+1) - u(: , i-1)) / (dt*2);
    ac(: , i) = (u(: , i+1) - 2 * u(: , i) + u(: , i-1)) / (dt^2);
end

G = tf([-M,0,0],[M,C,K]);                                 % single DOF CDM Transfer Function
uG = lsim(G , P ,[0.001:0.001:0.001*2000]);

plot(ac);
hold on