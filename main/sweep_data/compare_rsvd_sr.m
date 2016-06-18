function compare_rsvd_sr(Y,A,trainClass,targetClass,sweep_steps,save_path)

if nargin < 5
    step = floor(size(Y,2)/20);
    sweep_steps = step:step:5*step;
end
if nargin < 6
    save_path = 'tmp_linux_result.mat';
end


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
base_time = toc
%% RSVD param sweep
% do some RSVD
% k = [floor(size(Y,2))/10:floor(size(Y,2))/10:size(Y,2), size(Y,2)];
lambda = 1;%for now 
fprintf('***sweep the k of Q when doing Y ~ QR for Classification***\n')
k = sweep_steps;
Q_corrPerc = zeros(size(k));
Q_time_cost = zeros(size(k));
Q_X_diff = zeros(size(k));
Q_Y_diff = zeros(size(k));
for i = 1:length(k)
    tic;
    [Q, R] = rsvd_fixed_rank(Y,k(i));%kblock: increase step of Q's rank
    Q_Y_diff(i) = norm(Y-Q*R');
    T = SR(A, Q, lambda);
    X = T * R';
    Q_resultClass = ClassifySR(X,trainClass,Y,A);
%     [Q_resultClass, ~,~] = ClassifyQR(Q,R,Y,A,trainClass);
    Q_corrPerc(i) = sum(Q_resultClass==targetClass)/numTestSample*100;
    Q_time_cost(i) = toc;
    Q_X_diff(i) = norm(X-X0);
end
% Q_corrPerc(end) = corrPerc;
fprintf('***END***\n\n');

%% save data
save(save_path)