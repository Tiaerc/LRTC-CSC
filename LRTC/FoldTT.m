function [X] = FoldTT(X, dim, i)
% dim = circshift(dim, [1-i, 1-i]);
% X = shiftdim(reshape(X, dim), length(dim)+1-i);
X=reshape(X,dim);