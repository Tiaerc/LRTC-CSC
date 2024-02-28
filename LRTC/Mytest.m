clc;
clear;
close all;

%addpath('mylib/'); % rmpath('mylib/');
T = double(imread('house256.tiff')); % imshow(uint8(T))
% T=double(importdata('09true.mat'));
subplot(3,2,1);
imshow(uint8(T));
title('OriginPicture');
%T = imresize(T,0.5);
% I=double(importdata('05true.mat'));
I=double(imread('house256_test.tiff'));
Omega = (uint8(T)==uint8(I)); % imagesc(Omega)
Y = T;
Y(logical(1-Omega)) = 255;
subplot(3,2,2);
imshow(uint8(Y));
title('CorruptedPicture');
X=T;
alpha = [1, 1, 1e-3];
alpha = alpha / sum(alpha);

maxIter = 500;
epsilon = 1e-5;

% "X" returns the estimation, 
% "errList" returns the list of the relative difference of outputs of two neighbor iterations 
%% Simple LRTC-TT (solve the relaxed formulation, SiLRTC-TT in the paper)
% beta = 0.1*ones(1, ndims(T));
% [X_STTADMM, errList_STTADMM,NTT,errList2] = SiLRTCTTADMM(...
%     T,...                      % a tensor whose elements in Omega are used for estimating missing value
%     Y,...
%     beta,...                % the relaxation parameter. The larger, the closer to the original problem. See the function for definitions.
%     maxIter,...         % the maximum iterations
%     epsilon...            % the tolerance of the relative difference of outputs of two neighbor iterations 
%     );
% subplot(3,2,3);
% imshow(uint8(X_STTADMM));
% title('SiLRTC-TTADMM');
% subplot(3,2,4);
% imshow(uint8(NTT));
% title('SiLRTC-TTADMMN');
% h=subplot(3,2,5);
% plot(1:length(errList2), errList2, ':r', 'linewidth', 1.5); hold on;
%% Simple LRTC-TT (solve the relaxed formulation, SiLRTC-TT in the paper)
beta = 0.1*ones(1, ndims(T));
[X_STTADMM2, errList_STTADMM2,NTT2,errList22] = SiLRTCTTADMM2(...
    T,...                      % a tensor whose elements in Omega are used for estimating missing value
    Y,...
    Omega,...
    alpha,...
    beta,...                % the relaxation parameter. The larger, the closer to the original problem. See the function for definitions.
    maxIter,...         % the maximum iterations
    epsilon,...            % the tolerance of the relative difference of outputs of two neighbor iterations 
    X...
    );
subplot(3,2,3);
imshow(uint8(X_STTADMM2));
title('SiLRTC-TTADMM2');
subplot(3,2,4);
imshow(uint8(NTT2));
title('SiLRTC-TTADMMN2');
h=subplot(3,2,5);
plot(1:length(errList22), errList22, ':r', 'linewidth', 1.5); hold on;