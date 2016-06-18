%Demo cross validation
%%
clear;
clc;
repeat_time = 1;
runtime = 20;

fixed_rank = 50;
Q_corrPerc = zeros(runtime);
corrPerc = zeros(runtime,1);
%%
h = waitbar(0,'1','Name','RSVD Param scanning',...
    'CreateCancelBtn',...
    'setappdata(gcbf,''canceling'',1)');
setappdata(h,'canceling',0)
for i = 1:repeat_time
    
    load('/Users/Haru/Desktop/Key Lab/Data/Caltech101.mat')
    [Y,A,trainClass,targetClass] = preproMat(X,Y,0.9);
    
    numTestSample = size(Y,2);
    SCI = zeros(1,numTestSample);
    xp = mat2cell(zeros(1,numTestSample),1);
    qp = mat2cell(zeros(1,numTestSample),1);
    
    [resultClass, ~,~] = ClassifyQR(Y,[],[],A,trainClass,'no_waitbar');
    %%
    corrPerc(i) = sum(resultClass==targetClass)/numTestSample*100;
    
    
    % k = [floor(size(Y,2))/10:floor(size(Y,2))/10:size(Y,2), size(Y,2)];
    for j = 1:runtime
        [Q, R] = rsvd_fixed_rank(Y,fixed_rank);
        waitbar(((i-1)*runtime + j)/(runtime*repeat_time),h,['runtime: ' num2str((i-1)*runtime + j) '/' num2str(runtime*repeat_time)]);
        [Q_resultClass, ~,~] = ClassifyQR(Q,R,Y,A,trainClass,'no_waitbar');
        Q_corrPerc(i,j) = sum(Q_resultClass==targetClass)/numTestSample*100;
        if getappdata(h,'canceling')
            break;
        end
    end
    delete(h)
end

fprintf('average corrPerc: %.1f',mean(corrPerc))
fprintf('average Q_corrPerc: %.1f',mean(mean(Q_corrPerc)))