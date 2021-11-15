clear
clc
load('MCK5400.mat');                                        % import MCK data

dofs = 1                                                            % set DOFs
M = M(1:dofs,1:dofs);                                       % cut DOFs
C = C(1:dofs,1:dofs);
K = K(1:dofs,1:dofs);

dofs = length(M);
diagM = diag(1:dofs);
dt = 0.001;

alpha = 0.25;
beta = 0.5;
a0 = (1 / alpha / dt / dt);
a1 = (beta /alpha / dt);
a2 = (1 / alpha / dt);
a3 = (1 / 2 / alpha -1);
a4 = (beta / alpha - 1);
a5 = (dt / 2 * (beta / alpha -2));
a6 = (dt * (1 - beta));
a7 = (dt * beta);
Ke = K + a0 * M + a1 * C;

u = zeros(dofs , 2000);
v = zeros(dofs , 2000);
ac = zeros(dofs , 2000);

for i = 2 : 2000
    t = 1/2000*(i-2);
    P(i) = sin (2*pi*2*t);
    PP = -P(i)* diagM + M * (a0 * u(:,i-1) + a2 * v(:,i-1) + a3 * ac(:,i-1)) + C * (a1 * u(:,i-1) + a4 * v(:,i-1) + a5 * ac(:,i-1));
    u(:,i) = Ke \ PP;                            
    ac(: , i) = a0 * (u(: , i) - u(: , i-1)) - a3 * ac(: , i-1) - a2 * v(: , i-1);
    v(: , i)= v(: , i-1) + a6 * ac(: , i-1) + a7 * ac(: , i);
end

plot(ac);
hold on