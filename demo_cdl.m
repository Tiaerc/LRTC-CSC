clear;
clc;


% load training data
load('Data/train_data_fruit.mat');
% load('MRI.mat');
% train_data = (X(1:180,1:216,1));
% load('fltrain_girl.mat');
train_data= double(train_data_fruit);
tic

for num = [0.001]
% for num = [0.001,0.01,0.1,1,100]
% Filter input images and compute highpass images
npd = 16;
fltlmbd = 10;
[hl, hh] = lowpass(train_data, fltlmbd, npd);

% Construct initial dictionary
filter_size =16;
filter_num =32;

D0 = zeros(filter_size,filter_size,filter_num);
D0(:,:,:,:) = single(randn(filter_size,filter_size,filter_num));

% Set up cbpdndl parameters
lambda = num;
DLOpts = [];
DLOpts.Verbose = 1;
DLOpts.MaxMainIter = 500;
DLOpts.rho = 100 * lambda + 1;
DLOpts.sigma = size(train_data,4);
DLOpts.AutoRho = 1;
DLOpts.AutoRhoPeriod = 10;
DLOpts.AutoSigma = 1;
DLOpts.AutoSigmaPeriod = 10;
DLOpts.XRelaxParam = 1.8;
DLOpts.DRelaxParam = 1.8;

% Do dictionary learning
[D, X, DLOptsinf] = cbpdndl_gpu(D0, hh, lambda, DLOpts);
toc

% save filters
path1 = strcat('/home/ltc/CSC/Copy_of_PWLS-CSCGR-master/Filter/fruit_filtersize_',num2str(filter_size),'_num_',num2str(filter_num),'_lambda_',num2str(lambda),'.mat');
save(path1,'D');

end
% Display learned dictionary
figure;
imdisp(tiledict(D));

% Plot functional value evolution
figure;
plot(DLOptsinf.itstat(:,2));
xlabel('Iterations');
ylabel('Functional value');


