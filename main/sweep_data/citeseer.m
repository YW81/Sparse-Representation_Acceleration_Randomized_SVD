%CITESEER
addpath('../')
load('../../Data/Citeseer.mat')
[Y,A,trainClass,targetClass] = preproMat(X,Y,0.9);
clearvars -except Y A trainClass targetClass
step = floor(size(Y,2)/20);
sweep_steps = step:step:15*step;
compare_svd_sr(Y,A,trainClass,targetClass,sweep_steps,'citeseer_sweep.mat')
