% RSVD autorank w/ only Q,R output
% kblock: size of random samples block added at each update
% TOL: when to stop adding samples, when ||Q*Q'*A - A||_2 < TOL
function [Q,R] = rsvd_autorank(A,kblock,TOL)
tic
    m = size(A,1);
    n = size(A,2);

    mRand = randn(n,kblock);
    Y = A*mRand; % m \times n * n \times k = m \times k
    
    exit_loop = 0;
    h = waitbar(0,'1','Name','rsvd_version2_auto_rank2 Decomposing',...
        'CreateCancelBtn',...
        'setappdata(gcbf,''canceling'',1)');
    setappdata(h,'canceling',0)
    while exit_loop~=1
        %Q = qr(Y,0); % m \times k
        [Q,~]=qr(Y,0);
        norm_tol = norm(Q*Q'*A - A)/norm(A);
        if norm_tol > TOL
%             fprintf('norm_tol = %f; increase rank..\n', norm_tol);
            mRand = randn(n,kblock);
            waitbar(kblock/m,h,['current rank of Q: ' num2str(kblock) ' / ' num2str(m)]); 

            Y_new = A*mRand; % m \times n * n \times k = m \times k
            Y = [Y, Y_new];
        else
            exit_loop = 1;
        end
        if getappdata(h,'canceling')
        	exit_loop = 1;
        end
    end
    delete(h)
    [Q,~] = qr(Y,0); % m \times k
    R = A'*Q;
    fprintf('using Q of size %d*%d\n', size(Q,1), size(Q,2));
    fprintf('RSVD time is %.1f\n',toc);
end
