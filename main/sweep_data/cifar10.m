%BIOINFORMATIC
addpath('../')
load('../../Data/cifar-10-batches-mat/data_batch_1.mat')
[Y,A,trainClass,targetClass] = preproMat(double(data'),double(labels)+1,0.1);
clearvars -except Y A trainClass targetClass
step = floor(min(size(Y,1),size(Y,2))/20);
sweep_steps = step:step:15*step;
compare_svd_sr(Y,A,trainClass,targetClass,sweep_steps,'../../result/bioinformatic_sweep.mat')
