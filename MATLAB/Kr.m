% Kr-alpha method 
clear
clc
load('MCK5400.mat');                                         % import MCK data
dofs = 1                                                              % set DOFs
M = M(1:dofs,1:dofs);                                        % cut DOFs
C = C(1:dofs,1:dofs);
K = K(1:dofs,1:dofs);
dofs = length(M);
diagM = diag(1:dofs);
dt = 0.001;
gama = 7/6;
beta = 25/36;
af = 1 / 6;
am = -0.5;
u = zeros(dofs , 2000);
v = zeros(dofs , 2000);
ac = zeros(dofs , 2000);
MM = inv(M * (1 - am) + C * gama * dt * (1 - af) + K * beta * dt^2 * (1 - af));
P = zeros(1,2000);

for i = 2 : 2000
    t = 1/2000 * (i-2);
    P(i) = sin (2*pi*2*t);
    PP = -diagM * ((1 - af) * P(i) + af * P(i-1)) - (M * am + C * (1 - af) * dt * (1 - gama) + K * (1 - af) * dt^2 * (0.5 - beta)) * ac(: , i-1) - (C * (1 - af) + C * af + K * (1 - af) * dt) * v(: , i-1) - (K * (1 - af) + K * af) * u(: , i-1);
    ac(: , i) = MM * PP;
    v(: , i) = v(: , i-1) + dt * ((1 - gama) * ac(: , i-1) + gama * ac(: , i));
    u(: , i) = u(: , i-1) + dt * v(: , i-1) + (dt^2) * ((0.5 - beta) * ac(: , i-1) + beta * ac(: , i));
end

ZP = P;
Zac =ac;
Zu = u;

plot(ac);

