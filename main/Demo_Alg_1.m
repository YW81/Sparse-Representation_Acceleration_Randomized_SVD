%Test the Classify function

%% load small data
% X 512*700
% Y 700*7
% 512 features, 700 samples, 7 classes

% load('Data_Caltech_processed_Mar222016.mat')
load('/Users/Haru/Desktop/Key Lab/key/Data/Caltech101.mat')
[Y,A,trainClass,targetClass] = preproMat(X,Y,0.9);
clearvars -except Y A trainClass targetClass

%% load tiny data
% D 324 * 391
% G 391 * 1
% 324 features, 391 samples, 2 classes

load('/Users/Haru/Desktop/Key Lab/Data/Hotspot.mat')
[Y,A,trainClass,targetClass] = preproMat(D,G,0.9);
clearvars -except Y A trainClass targetClass

%% load relatively small data
load('/Users/Haru/Desktop/Key Lab/Data/chineseposts.mat')
[Y,A,trainClass,targetClass] = preproMat(Y,C,0.9);
clearvars -except Y A trainClass targetClass

%% load medium data
% X = 3705 * 3312
% Y = 3312 * 6
% 3705 features, 3312 samples, 6 classes

load('/Users/Haru/Desktop/Key Lab/key/Data/Citeseer.mat')
[Y,A,trainClass,targetClass] = preproMat(X,Y,0.9);
clearvars -except Y A trainClass targetClass

%% load big data
% X = 4196 * 4196
% Y = 4196 * 4
% 4196 features, 4196 samples, 4 classes


load('/Users/Haru/Desktop/Key Lab/key/Data/WEBKB4.mat')
[Y,A,trainClass,targetClass] = preproMat(X,Y,0.9);
clearvars -except Y A trainClass targetClass

%% load lengthy data
% X 21531 * 600
% Y 600 * 6
load('/Users/Haru/Desktop/Key Lab/Data/Reuters.mat')
[Y,A,trainClass,targetClass] = preproMat(X,Y,0.9);
clearvars -except Y A trainClass targetClass


%% load colossal data
% X 19938*19938
% Y 19938*20
% 19938 features, 19938 samples, 20 classes

load('/Users/Haru/Desktop/Key Lab/Data/20News.mat')
[Y,A,trainClass,targetClass] = preproMat(X,Y,0.9999);
clearvars -except Y A trainClass targetClass

%% load weibo data
%bad data
load('/Users/Haru/Desktop/Key Lab/key/Data/Weibo/weibo.mat')
[Y,A,trainClass,targetClass] = preproMat(X,Y,0.9);
clearvars -except Y A trainClass targetClass

%% load bioinfo
load('/Users/Haru/Desktop/Key Lab/key/Data/bioinformatics.mat')
X = [A,Y];
Y = [C;groups];
[Y,A,trainClass,targetClass] = preproMat(X,Y,0.9);
clearvars -except Y A trainClass targetClass

%% load cifar 10
load('/Users/Haru/Desktop/Key Lab/key/Data/cifar-10-batches-mat/data_batch_1.mat')
[Y,A,trainClass,targetClass] = preproMat(double(data'),double(labels)+1,0.01);
clearvars -except Y A trainClass targetClass

%% Classification using SR from Gao Longwen
% 
tic
lambda = 1;%for now
numTestSample = size(Y,2);
X0 = SR(A, Y, lambda);
resultClass = ClassifySR(X0,trainClass,Y,A);
corrPerc = sum(resultClass==targetClass)/numTestSample*100;
fprintf('Correct%% = %.1f%%\n',corrPerc);
fprintf('***END***\n\n');  
toc

%% visualize result
figure1 = figure;
subplot1 = subplot(1,1,1,'Parent', figure1);
hold on
plot(resultClass,'*','Parent',subplot1);
plot2 = plot(targetClass,'^','Parent',subplot1);

hold off

%% Q via RSVD fixed rank
% do some RSVD
fprintf('***Using fixed rank Y ~ QR for Classification***\n')

fixed_rank = 50;
str_Classify_condition = sprintf('fixed rank = %d',fixed_rank);
% [U,S,V] = rsvd_version2(Y,fixed_rank);
% Q = U*S;
% R = V;

[Q, R] = rsvd_fixed_rank(Y,fixed_rank);%kblock: increase step of Q's rank
%% Q via RSVD auto rank
% do some RSVD
fprintf('***Using auto rank Y ~ QR for Classification***\n')
tolerance = 0.05;
str_Classify_condition = sprintf('autorank tolerance = %d',tolerance);

[Q, R] = rsvd_autorank(Y,1,tolerance);%kblock: increase step of Q's rank

%% Classify through Q
tic
lambda = 1;%for now 
numTestSample = size(Y,2);
T = SR(A, Q, lambda);
X = T * R';
Q_resultClass = ClassifySR(X,trainClass,Y,A);
Q_corrPerc = sum(Q_resultClass==targetClass)/numTestSample*100;
fprintf('When colums of Y is compressed by %.1f%% with %s, Correct%% = %.1f%%\n',(1-size(Q,2)/size(Y,2))*100, str_Classify_condition, Q_corrPerc);
fprintf('worse for %.1f%%\n',corrPerc-Q_corrPerc);
fprintf('***END***\n\n');
t = toc
%% RSVD param sweep
% do some RSVD
fprintf('***sweep the k of Q when doing Y ~ QR for Classification***\n')

% k = [floor(size(Y,2))/10:floor(size(Y,2))/10:size(Y,2), size(Y,2)];
lambda = 1;%for now 

% k_step = 3;
% k = 1:k_step:size(Y,2);
k = [5:5:30];
% k = 150::150
Q_corrPerc = zeros(size(k));
Q_time_cost = zeros(size(k));
Q_X_diff = zeros(size(k));
Q_Y_diff = zeros(size(k));
h = waitbar(0,'1','Name','RSVD Param scanning',...
    'CreateCancelBtn',...
    'setappdata(gcbf,''canceling'',1)');
setappdata(h,'canceling',0)
for i = 1:length(k)
    tic;
    [Q, R] = rsvd_fixed_rank(Y,k(i));%kblock: increase step of Q's rank
    Q_Y_diff(i) = norm(Y-Q*R');
    waitbar(k(i)/size(Y,2),h,['rank of Q: ' num2str(k(i)) '/' num2str(size(Y,2))]);
    if getappdata(h,'canceling')
        break;
    end
    T = SR(A, Q, lambda);
    X = T * R';
    Q_resultClass = ClassifySR(X,trainClass,Y,A);
%     [Q_resultClass, ~,~] = ClassifyQR(Q,R,Y,A,trainClass);
    Q_corrPerc(i) = sum(Q_resultClass==targetClass)/numTestSample*100;
    Q_time_cost(i) = toc;
    Q_X_diff(i) = norm(X-X0);
end

delete(h)
% Q_corrPerc(end) = corrPerc;
fprintf('***END***\n\n');

%% visualize param sweep result
figure1 = figure;

% Create subplot
subplot1 = subplot(3,1,1,'Parent',figure1,'YGrid','on','XGrid','on');
box(subplot1,'on');
hold(subplot1,'all');

% Create multiple lines using matrix input to plot
plot1 = plot(k,[Q_corrPerc',corrPerc*ones(size(Q_corrPerc))']);
xlabel(subplot1,'rank of Q')
ylabel(subplot1,'percentage of correct (%)')
set(plot1(1),'DisplayName','Q');
set(plot1(2),'DisplayName','A');
legend1 = legend(gca,'show');
set(legend1,'Location','Best');

set(legend1,'FontSize',9);

% time cost
subplot2 = subplot(3,1,2,'Parent',figure1,'YGrid','on','XGrid','on');
box(subplot2,'on');
hold(subplot2,'all');
plot2 = plot(k,Q_time_cost);
set(plot2,'DisplayName','Time cost');
% Create legend
legend2 = legend(subplot2,'show');
set(legend2,'Location','Best');
title('Time cost')
xlabel(subplot2,'rank of Q')
ylabel(subplot2, 'Time cost(seconds)');

% X difference
subplot3 = subplot(3,1,3,'Parent',figure1,'YGrid','on','XGrid','on');
box(subplot3,'on');
hold(subplot3,'all');
hAx3 = plotyy(k,Q_X_diff,k,Q_Y_diff);
% Create legend
title('Difference of matrices')
xlabel(subplot3,'rank of Q') 
ylabel(hAx3(1), 'Difference between X and X_0');
ylabel(hAx3(2), 'Difference between Y and QR');

linkaxes([subplot1,subplot2,subplot3],'x')

%% RSVD param sweep without need of base
numTestSample = size(Y,2);
% do some RSVD
fprintf('***sweep the k of Q when doing Y ~ QR for Classification***\n')

% k = [floor(size(Y,2))/10:floor(size(Y,2))/10:size(Y,2), size(Y,2)];
lambda = 1;%for now 

% k_step = 3;
% k = 1:k_step:size(Y,2);
k = [1:5:150];
% k = 150::150
Q_corrPerc = zeros(size(k));
Q_time_cost = zeros(size(k));
h = waitbar(0,'1','Name','RSVD Param scanning',...
    'CreateCancelBtn',...
    'setappdata(gcbf,''canceling'',1)');
setappdata(h,'canceling',0)
for i = 1:length(k)
    tic;
    [Q, R] = rsvd_fixed_rank(Y,k(i));%kblock: increase step of Q's rank
    waitbar(k(i)/size(Y,2),h,['rank of Q: ' num2str(k(i)) '/' num2str(size(Y,2))]);
    if getappdata(h,'canceling')
        break;
    end
    T = SR(A, Q, lambda);
    X = T * R';
    Q_resultClass = ClassifySR(X,trainClass,Y,A);
%     [Q_resultClass, ~,~] = ClassifyQR(Q,R,Y,A,trainClass);
    Q_corrPerc(i) = sum(Q_resultClass==targetClass)/numTestSample*100;
    Q_time_cost(i) = toc;
end

delete(h)
% Q_corrPerc(end) = corrPerc;
fprintf('***END***\n\n');

%% visualize param sweep result without need of base
figure1 = figure;

% Create subplot
subplot1 = subplot(2,1,1,'Parent',figure1,'YGrid','on','XGrid','on');
box(subplot1,'on');
hold(subplot1,'all');

% Create multiple lines using matrix input to plot
plot1 = plot(k,Q_corrPerc');
xlabel(subplot1,'rank of Q')
ylabel(subplot1,'percentage of correct (%)')
set(plot1(1),'DisplayName','Q');
legend1 = legend(gca,'show');
set(legend1,'Location','Best');

set(legend1,'FontSize',9);

% time cost
subplot2 = subplot(2,1,2,'Parent',figure1,'YGrid','on','XGrid','on');
box(subplot2,'on');
hold(subplot2,'all');
plot2 = plot(k,Q_time_cost);
set(plot2,'DisplayName','Time cost');
% Create legend
legend2 = legend(subplot2,'show');
set(legend2,'Location','Best');
title('Time cost')
xlabel(subplot2,'rank of Q')
ylabel(subplot2, 'Time cost(seconds)');

linkaxes([subplot1,subplot2],'x')


%% RSVD param sweep parfor version
% do some RSVD
fprintf('***sweep the k of Q when doing Y ~ QR for Classification***\n')

% k = [floor(size(Y,2))/10:floor(size(Y,2))/10:size(Y,2), size(Y,2)];
lambda = 1;%for now 

% k_step = 3;
% k = 1:k_step:size(Y,2);
k = [1:10:300];
% k = 150::150
Q_corrPerc = zeros(size(k));
Q_time_cost = zeros(size(k));
Q_X_diff = zeros(size(k));
Q_Y_diff = zeros(size(k));
Q_resultClass = zeros(size(k));
parfor i = 1:length(k)
    tic;
    [Q, R] = rsvd_fixed_rank(Y,k(i));%kblock: increase step of Q's rank
    Q_Y_diff(i) = norm(Y-Q*R');
    T = SR(A, Q, lambda);
    X = T * R';
    Q_resultClass(i) = ClassifySR(X,trainClass,Y,A);
%     [Q_resultClass, ~,~] = ClassifyQR(Q,R,Y,A,trainClass);
    Q_corrPerc(i) = sum(Q_resultClass(i)==targetClass)/numTestSample*100;
    Q_time_cost(i) = toc;
    Q_X_diff(i) = norm(X-X0);
    print('finish ' + k(i))
end

delete(h)
% Q_corrPerc(end) = corrPerc;
fprintf('***END***\n\n');

%% RSVD stability check
fprintf('***check the stability when doing Y ~ QR for Classification***\n')

% do some RSVD
tolerance = 0.1;
runtime = 5;

h = waitbar(0,'1','Name','RSVD Stability checking',...
    'CreateCancelBtn',...
    'setappdata(gcbf,''canceling'',1)');
setappdata(h,'canceling',0)
for i = 1:runtime
    waitbar(i/runtime,h,['Iteration: ' num2str(i) '/' num2str(runtime)]);
    if getappdata(h,'canceling')
        break;
    end
[Q, R] = rsvd_autorank(Y,1,tolerance);
[Q_resultClass, ~,~] = ClassifyQR(Q,R,Y,A,trainClass);
Q_corrPerc(i) = sum(Q_resultClass==targetClass)/numTestSample*100;
end
delete(h)
E_corr = mean(Q_corrPerc);
fprintf('E(correct%%) = %.1f%%, worse for %.1f%%\n', E_corr, corrPerc-E_corr);
fprintf('Var(correct%%) = %.1f\n', var(Q_corrPerc));
fprintf('Winning Rate = %.1f%%\n',sum(Q_corrPerc>corrPerc)/runtime*100); 
fprintf('***END***\n\n');

%% visualize param sweep result
figure1 = figure;
axes1 = axes('Parent',figure1);
hold(axes1,'on');
plot1 = plot([Q_corrPerc',corrPerc*ones(size(Q_corrPerc))',E_corr*ones(size(Q_corrPerc))']);
xlabel('times of iteration')
ylabel('percentage of correct (%)')
set(plot1(1),'DisplayName','tol = 0.1');
set(plot1(2),'DisplayName','tol = 0');
set(plot1(3),'DisplayName','E(tol = 0)');
legend1 = legend(axes1,'show');
set(legend1,'FontSize',9);

%% visualize result

figure2 = figure;
subplot2 = subplot(1,1,1,'Parent', figure2);
hold on

hAx3 = plot(resultClass,'r*','Parent',subplot2);
plot5 = plot(targetClass,'^','Parent',subplot2);
plot4 = plot(Q_resultClass,'b*','Parent',subplot2);
set(hAx3,'DisplayName','A resultClass');
set(plot5,'DisplayName','targetClass');
set(plot4,'DisplayName','Q resultClass');

legend2 = legend(gca,'show');
set(legend2,'FontSize',9);
xlabel('Sample Index')
ylabel('Class Index')
hold off

%% Debug: 
fprintf('***Debugging Classification***\n')
[resultClass, ~,~] = ClassifyQR(Y(:,1),[],[],A,trainClass);
corrPerc = (resultClass==targetClass(1))*100;
fprintf('Correct%% = %.1f%%\n',corrPerc);
fprintf('***END***\n\n');  

%% Original Classification program (for debug)

% [resultClass, SCI,xp] = Classify(y);

%% Classification using Classify method in Face Recog

% tic
% for i = 1:numTestSample
%     %resultClass: classification result of c
%     %SCI: Sparsity concentration index in [0,1]. 1 if identical with a dictionary
%     %unit, 0 if its sparse coeffs spread evenly over all classes
%     [resultClass(i), SCI(i),xp{i}] = Classify(Y(:,i));
% end
% toc
% 
% X = cell2mat(xp);%the solution of Y = AX
% corrPerc = sum(resultClass==targetClass)/numTestSample

%% init
% resultClass = zeros(1,numTestSample);
% SCI = zeros(1,numTestSample);
% xp = mat2cell(zeros(1,numTestSample),1);
% qp = mat2cell(zeros(1,numTestSample),1);
% 
%% Classification using Classify method in Face Recog but with code in ClassifyQR
% 
% fprintf('***Using Y for Classification***\n')
% [resultClass, ~,~] = ClassifyQR(Y,[],[],A,trainClass);
% corrPerc = sum(resultClass==targetClass)/numTestSample*100;
% fprintf('Correct%% = %.1f%%\n',corrPerc);
% fprintf('***END***\n\n');   



