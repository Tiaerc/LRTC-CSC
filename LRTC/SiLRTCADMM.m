%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% % Simple Low Rank Tensor Completion (SiLRTC) 
% Time: 03/11/2012
% Reference: "Tensor Completion for Estimating Missing Values 
% in Visual Data", PAMI, 2012.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [X, errList, N,errList2] = SiLRTCADMM(T, Y, Omega, alpha, gamma, maxIter, epsilon, X)
%%%%%%%%%%%%%%%%%%%%%%%%%%
% min(X, M1, M2, M3,... Mn): (\gamma1||X_(1)-M1||^2 + \gamma2||X_(2)-T2||^2 + \gamma3||X_(3)-T3||^2 + ...)/2 + 
%               \alpha1||M1||_* + \alpha2||M2||_* + \alpha3||M3||_* + ....
%         s.t.  X_\Omega = T_\Omega
%%%%%%%%%%%%%%%%%%%%%%%%%%
if nargin < 7
    X = T;
    X(logical(1-Omega)) = mean(T(Omega));
end
N = full(sptenrand([256 256 3],10000)*256);
% N=(rand(256,256,3)*256);
% V=(rand(256,256,3)*256);
V=zeros(256,256,3);
errList = zeros(maxIter, 1);
errList2 = zeros(maxIter, 1);
normT = norm(T(:));
%L = errList;
dim = size(T);
M = cell(ndims(T), 1);
gammasum = sum(gamma);
tau = alpha./ gamma;
I=T;
I(I<300)=0;
Omega22=(I==ones(256,256,3));
c=0.5;
p=0.2;
%normT = norm(T(:));
for k = 1:maxIter         
       
    if mod(k, 50) == 0
        fprintf('SiLRTC-TTADMM: iterations = %d   difference=%f\n', k, errList2(k-1));
    end
    %更新X^k+1/2
    Xsum = 0;
    for i = 1:ndims(T)
        M{i} = Fold(Pro2TraceNorm(Unfold(X, dim, i), tau(i)), dim, i);
        Xsum = Xsum + gamma(i) * M{i};
    end 
       Xlast = X; 
        X = Xsum / gammasum;
       %更新X^k+1
        X=(Y-N)-V/c;
        X=double(X);
        Omega3 =(uint8(T(logical(1-Omega)))==uint8(X(logical(1-Omega))));
        if(~isequal(Omega3,Omega22))
            X(Omega3)=T(Omega3);        
        end
        X(Omega) = T(Omega);
        Omega=(uint8(T)==uint8(X));
        N=softthresholding(Y-X-V/c,p/c);
        V=V+c*(X+N-Y);
        errList2(k) = norm(X(:)-T(:));
    errList(k) = norm(X(:)-Xlast(:));
%     if (k>1 && (errList(k) < epsilon || errList(k)>errList(k-1)))
%         errList = errList(1:k);
%         break;
%     end
    %L(k) = norm(X(:)-T(:)) / normT;
end

% for i = 1:200
%         Xllast=X;
%         X=(double(Y)-N)-V/c;
%         X=double(X);
%         X(Omega) = T(Omega);
%         N=softthresholding(double(Y)-X-V/c,p/c);
%         V=V+c*(X+N-double(Y));
%         errList2(k) = norm(X(:)-T(:)) / normT;
%         if(norm(Xllast(:)-X(:))/ normT<1e-7)
%             disp('sssssss');
%             break;
%         end
% end
fprintf('SiLRTC-TTADMM ends: total iterations = %d   difference=%f\n\n', k, errList(k));
end

function [ soft_thresh ] = softthresholding( b,lambda )
    soft_thresh = sign(b).*max(abs(b) - lambda,0);
end
