function [D] = pdist2(X)
%PDIST2(X) calculates squared pairwise distance
%   X: data matrix, each row is an instance and each column is a feature

    X2 = sum(X .* X, 2);
    XX = X * X';
    D = X2 + X2' - 2 * XX;
    D = D - min(min(D));
end

