function [resultClass, SCI,X] = ClassifyQR(Q,R,Y,A,trainClass,noNeedWaitbar)
% Classification with Y approx QR
% waitbar: empty if needed
RejectThreshold = 0;
lambda = 1000;
szQ = size(Q);
if isempty(R)
    R = eye(szQ(2));
end
qp = mat2cell(zeros(1,szQ(2)),1);

%% Solve T for eq Q = AT



if nargin < 6 % isempty needWaitbar
    h = waitbar(0,'1','Name','Solve T for eq Q = AT',...
        'CreateCancelBtn',...
        'setappdata(gcbf,''canceling'',1)');
    setappdata(h,'canceling',0)
    tic
    for i = 1:szQ(2)
        waitbar(i/szQ(2),h,['#solved column: ' num2str(i) '/' num2str(szQ(2))]);
        
        q = Q(:,i);
        q0 = A'*q; %bo
        qp{i} = l1eq_pd(q0, A, [], q, 5e-2, lambda);
        
        if getappdata(h,'canceling')
            break;
        end
    end
    delete(h)
else
    tic
    for i = 1:szQ(2)
        q = Q(:,i);
        q0 = A'*q; %bo
        qp{i} = l1eq_pd(q0, A, [], q, 5e-2, lambda);
    end
end

t = toc;
fprintf('Solve T for equation Q = AT time is %.1f\n',t);
T = cell2mat(qp);

%% return to Y = AX

X = T * R';
if isempty(Y)
    Y = Q * R';
end


%% Classification

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

end
%
% function [CVector] = GetClassVector(leftIndex, rightIndex, orgVector)
% CVector = zeros(size(orgVector,1), 1);
% CVector(leftIndex:rightIndex, 1) = orgVector(leftIndex:rightIndex, 1);
% end

