function resultClass = ClassifyFSR(X,trainClass,Y,D,A)


RejectThreshold = 0;
szX = size(X);
SCI = zeros(1,szX(2));
resultClass = zeros(1,szX(2));
for i = 1:szX(2)
    
    
    
    residuals = zeros(1,numClass);
    for iClass = 1: numClass
        xpClass = xp;
        xpClass(trainLabel~= iClass) = 0;
        residuals(iClass) = norm(Y(:,i) - A*xpClass);
    end
    [val, ind] = min(residuals);
    if(ind==testLabel(i))
        correctSample = correctSample+1;
    end
    

end
