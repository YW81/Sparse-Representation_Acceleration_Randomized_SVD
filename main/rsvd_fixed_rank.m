% RSVD fixed rank
% Only output Q instead of U Sigma V
function [Q,R] = rsvd_fixed_rank(A,k)
    m = size(A,1);
    n = size(A,2);

    mRand = randn(n,k);
    Y = A*mRand; % m \times n * n \times k = m \times k

    [Q,~] = qr(Y,0); % m \times k
    R = A'*Q;

end
