%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% % Simple Low Rank Tensor Completion (SiLRTC) 
% Time: 03/11/2012
% Reference: "Tensor Completion for Estimating Missing Values 
% in Visual Data", PAMI, 2012.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [X, errList, N,errList2] = SiLRTCTTADMM2(T, Y, Omega, alpha, gamma, maxIter, epsilon, X)
%%%%%%%%%%%%%%%%%%%%%%%%%%
% min(X, M1, M2, M3,... Mn): (\gamma1||X_(1)-M1||^2 + \gamma2||X_(2)-T2||^2 + \gamma3||X_(3)-T3||^2 + ...)/2 + 
%               \alpha1||M1||_* + \alpha2||M2||_* + \alpha3||M3||_* + ....
%         s.t.  X_\Omega = T_\Omega
%%%%%%%%%%%%%%%%%%%%%%%%%%
if nargin < 7
    X = T;
    X(logical(1-Omega)) = mean(T(Omega));
end
N = full(sptenrand([256 256 3],20000)*256);
V=(rand(256,256,3)*256);
% V=zeros(256,256,3);
errList = zeros(maxIter, 1);
errList2 = zeros(maxIter, 1);
normT = norm(T(:));
%L = errList;
dim = size(T);
M = cell(ndims(T), 1);
gammasum = sum(gamma);
tau = alpha./ gamma;

c=1;
sigma=10;
p=0.05;
%normT = norm(T(:));
for k = 1:maxIter         
           Xlast = X; 
    if mod(k, 50) == 0
        fprintf('SiLRTC-TTADMM: iterations = %d   difference=%f\n', k, errList2(k-1));
    end
    %更新X^k+1/2
    Xsum = 0;
    for i = 1:ndims(T)
        M{i} = FoldTT(Pro2TraceNorm(UnfoldTT(X, dim, i), tau(i)), dim, i);
        Xsum = Xsum + gamma(i) * M{i};
    end 
        X = Xsum / gammasum;       

        N=double(N);
        %展开
        XX=UnfoldTT(X,dim,1);
        NN=UnfoldTT(N,dim,1);
        YY=UnfoldTT(Y,dim,1);
        VV=UnfoldTT(V,dim,1);
        XX=((YY-NN)*c)/(sigma+c)-VV/c+(sigma/(sigma+c))*XX;
        X=FoldTT(XX,dim,1);
        X(Omega) = T(Omega);
        XX=UnfoldTT(X,dim,1);
        NN=softthresholding(YY-XX-VV./c,p/c);
        VV=VV+c.*(XX+NN-YY);
        N=FoldTT(NN,dim,1);
        V=FoldTT(VV,dim,1);
        errList2(k) = norm(X(:)-T(:))/normT;
        errList(k) = norm(X(:)-Xlast(:))/normT;
    if (k>1 && (errList(k) < epsilon))
        errList = errList(1:k);
        break;
    end
    %L(k) = norm(X(:)-T(:)) / normT;
end

fprintf('SiLRTC-TTADMM2 ends: total iterations = %d   difference=%f\n\n', k, errList(k));
end

function [ soft_thresh ] = softthresholding( b,lambda )
    soft_thresh = sign(b).*max(abs(b) - lambda,0);
end
