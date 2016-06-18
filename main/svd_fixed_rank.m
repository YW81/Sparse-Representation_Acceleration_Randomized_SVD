% SVD fixed rank
% Only output Q instead of U Sigma V
function [Q,R] = svd_fixed_rank(A,k)
    [U,S,V] = svd(A,0);
    Q = U(:,1:k)*S(1:k,1:k);
    R = V(:,1:k);
    

end