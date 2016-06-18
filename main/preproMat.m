function [Y,A,trainClass,targetClass] = preproMat(mData,mClass,rTrain)
% mData: #features * #samples
% mClass: #samples * #classes
% rTrain: ratio of training data
if isempty(rTrain)
    rTrain = 0.8;
end



% nClass = size(mClass,2);
ind = randperm(size(mData,2));
l = length(ind);
nTrain = floor(l*rTrain);% number of training data
indTrain = ind(1:nTrain);
indTarget = ind(nTrain+1:end);

A = mData(:,indTrain);
Y = mData(:,indTarget);

if size(mClass,2) > 1
    trainClass = mClass(indTrain,:);
    targetClass = mClass(indTarget,:);
    for i = 1:size(targetClass,2)
        targetClass(:,i)=targetClass(:,i)*i;
    end
    targetClass = sum(targetClass,2)';
else
    targetClass = mClass(indTarget,:)';
    array_trainClass = mClass(indTrain,:);
    trainClass = zeros(size(mClass,1),max(mClass));
    for i = 1:size(array_trainClass,1)
        trainClass(i,array_trainClass(i))=1;
    end
end
    

