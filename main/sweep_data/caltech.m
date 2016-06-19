%CITESEER
addpath('../')
load('../../Data/Caltech101.mat')
[Y,A,trainClass,targetClass] = preproMat(X,Y,0.9);
clearvars -except Y A trainClass targetClass
step = floor(min(size(Y,2), size(Y,1))/20);
sweep_steps = step:step:15*step;
compare_svd_sr(Y,A,trainClass,targetClass,sweep_steps,'../../result/caltech_sweep.mat')
