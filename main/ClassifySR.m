function resultClass = ClassifySR(X,trainClass,Y,A)


RejectThreshold = 0;
szX = size(X);
SCI = zeros(1,szX(2));
resultClass = zeros(1,szX(2));
for i = 1:szX(2)
    vMax = 0;
    jMax = 0;
    MaxVector = 0;
    
    for j = 1: size(trainClass,2)
        %         CVector = GetClassVector((j-1) * TrainSize + 1, j * TrainSize, X(:,i));
        CIndex = find(trainClass(:,j)==1);
        CVector = zeros(szX(1),1);
        CVector(CIndex) = X(CIndex,i);
        y0 = A * CVector;
        val = abs(1 /(pdist([Y(:,i) y0]')));
        if(vMax < val)
            vMax = val;
            jMax = j;
        end
        if(sum(abs(MaxVector(:,1))) < sum(abs(CVector(:,1))))
            MaxVector = CVector;
        end
    end
    
    %compute SCI
    SCI(i) = (size(trainClass,2) * sum(abs(MaxVector(:,1))) / sum(abs(X(:,i))) - 1) / (size(trainClass,2) - 1);
    if(SCI(i) < RejectThreshold)
        resultClass(i) = -1;
    else
        resultClass(i) = jMax;
    end
end
