%BIOINFORMATIC
addpath('../')
load('../../Data/bioinformatics.mat')
X = [A,Y];
Y = [C;groups];
[Y,A,trainClass,targetClass] = preproMat(X,Y,0.9);
clearvars -except Y A trainClass targetClass
step = floor(size(Y,2)/20);
sweep_steps = step:step:5*step;
compare_rsvd_sr(Y,A,trainClass,targetClass,[],'bioinformatic_sweep.mat')
