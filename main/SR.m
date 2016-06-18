function X = SR(A, Y, lambda, L)
%UGSR Group Sparse Representation solver
%   -Input:
%       A: an n*m dictionary formed by m vectors of dimension n
%       Y: a test matrix of dimension d*n, to be represented as a combination
%   of vectors in A
%       lambda: coefficient of sparsity regularizer
%       L: largest eigenvalue of A'A
%   -Output:
%       X: output coefficient matrix

% parameter settings
bound = 1e-4;

if nargin < 5,
    E = eig(A' * A);
    L = E(end);
    clear E;
end

Lb = 1 / L;
m = size(A, 2);
n = size(Y, 2);
X = zeros(m, n);
D = Lb * (A' * A);

for iy = 1 : n
    % initialize values
    F = Lb * A' * Y(:, iy);
    x = zeros(m, 1);
    x0 = x;
    residual = Inf;

    % iteration
    while residual > bound
        % proximal operator
        for i = 1 : m
            x(i) = max(0, 1 - lambda * Lb / abs(x(i))) * x(i);
        end
        % gradient descent iteration for x
        x = x0 + F - D * x;
        % compute the residual error and data save
        residual = norm(x - x0);
        x0 = x;
    end
    
    % output result
    X(:, iy) = x;
end
end
