%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% % Simple Low Rank Tensor Completion (SiLRTC) 
% Time: 03/11/2012
% Reference: "Tensor Completion for Estimating Missing Values 
% in Visual Data", PAMI, 2012.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [X, errList, N,errList2] = SiLRTCTTADMM(T, Y, gamma, maxIter, epsilon)
%%%%%%%%%%%%%%%%%%%%%%%%%%
% min(X, M1, M2, M3,... Mn): (\gamma1||X_(1)-M1||^2 + \gamma2||X_(2)-T2||^2 + \gamma3||X_(3)-T3||^2 + ...)/2 + 
%               \alpha1||M1||_* + \alpha2||M2||_* + \alpha3||M3||_* + ....
%         s.t.  X_\Omega = T_\Omega
%%%%%%%%%%%%%%%%%%%%%%%%%%
% if nargin < 7
%     X = T;
%     X(logical(1-Omega)) = mean(T(Omega));
% end
X=zeros(256,256,3);
N = full(sptenrand([256 256 3],20000)*256);
A=Y-N;
A=double(A);
V=zeros(256,256,3);
errList = zeros(maxIter, 1);
errList2 = zeros(maxIter, 1);
normT = norm(T(:));
%L = errList;
dim = size(T);
M = cell(ndims(T), 1);
Q = cell(ndims(T), 1);
alpha = [1, 1, 1e-3];
alpha = alpha / sum(alpha);

gammasum = sum(gamma);

p=0.05;
lambda=0.01;
c=0.03;

for i = 1:ndims(N)
    Q{i}=UnfoldTT(double(N),dim,i);
end

%normT = norm(T(:));
for k = 1:maxIter         
       
    if mod(k, 20) == 0
        fprintf('SiLRTC-TTADMM: iterations = %d   difference=%f\n', k, errList2(k-1));
    end
    
    tau = alpha./ p;
    for i=ndims(T)
        alpha(i)=1./(2*rank(UnfoldTT(X,dim,i))+0.0001);
    end
    alpha = alpha / sum(alpha);

    for i = 1:ndims(T)
        M{i} = Pro2TraceNorm(UnfoldTT(X, dim, i)+Q{i}/p, tau(i));
    end    

    Xsum=0;
    for i=1:ndims(T)
        Xsum=Xsum+FoldTT((M{i}-Q{i}/p),dim,i);
    end
    X=((A-V./p)+Xsum/ndims(T))/2;
    
%     Xsum = 0;
%     for i = 1:ndims(T)
%         M{i} = FoldTT(Pro2TraceNorm(UnfoldTT(X, dim, i), tau(i)), dim, i);
%         Xsum = Xsum + gamma(i) * M{i};
%     end 
%        Xlast = X; 
%        X = Xsum / gammasum;
 
        A=softthresholding(Y-X-V/p,lambda/p);
  
        V=V+c*(X-A);
        for i = 1:ndims(T)
            Q{i}=Q{i}+c*(UnfoldTT(X,dim,i)-M{i});
        end
        
        errList2(k) = norm(X(:)-T(:))/normT;
         errList(k) = norm(X(:)-A(:))/normT;
    if (k>1 && (errList2(k) < epsilon))
        errList = errList(1:k);
        break;
    end
    %L(k) = norm(X(:)-T(:)) / normT;
end
N=Y-A;
disp(alpha);
fprintf('SiLRTC-TTADMM ends: total iterations = %d   difference=%f\n\n', k, errList2(k));
end

function [ soft_thresh ] = softthresholding( b,lambda )
    soft_thresh = sign(b).*max(abs(b) - lambda,0);
end
