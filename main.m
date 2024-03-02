% clc;
clear;
close all;
cur = cd;
addpath(genpath(cur));

original_image = imread('/home/CSC/data/Peppers.jpg');
original_image = double(original_image);

alpha = [3, 3, 1];
alpha = alpha / sum(alpha);
epsilon = 1e-6;


imageName = 'peppers';
[rows, cols] = size(original_image);

img_size = rows;
pro_views = 64;
bins = 512;


pwls_iter = 20;
pwls = zeros(size(original_image));
reconstruction = zeros(size(original_image));
mr = 0.7;

dim = size(original_image);
Nway = size(original_image);

P = round((1-mr)*prod(Nway));%round 四舍五入为最小整数
Known = randsample(prod(Nway),P);%从1到prod(Nway)中随机抽取P个数
[Known,~] = sort(Known);
Omega = zeros(dim);
Omega(Known) = 1;
Omega = logical(Omega);%换成逻辑值

 gpuDevice()

for filter_lambda = 10
    
    % load the filter
    filter_size = 16;
    filter_num = 32;
    % filter_lambda =10;
    filter = strcat('fruit_filtersize_',num2str(filter_size), ...
        '_num_',num2str(filter_num),...
        '_lambda_',num2str(filter_lambda));
    load(strcat('Filter/',filter,'.mat'));
    
    % params of cscgr
    lambda = 3.8;
    rho = 100 * lambda + 1;
    tau = 0;
    iter = 500;

   
     
    % create folder to save the results
    filename = strcat(num2str(pro_views),'_mr_',num2str(mr),'_lambda_',num2str(lambda),'_rho_',num2str(rho),'_mu_',num2str(tau));
    path = strcat('/home/ltc/CSC/Result/',filter,'/',imageName,'/',filename);
    if ~exist(path,'dir')
        mkdir(path);
    end
    % the file used to record the psnrs
    if exist(strcat(path,'/result.txt'),'file')
        delete(strcat(path,'/result.txt'));
    end
    result = fopen(strcat(path,'/result.txt'),'a+');

    % save filter
    save(strcat(path,'/',filter,'.mat'),'D');

    
    
    X_S = original_image;
    Observed = original_image;
    Xlast = zeros(size(original_image));
    Observed(logical(1-Omega)) = 0;
    X_S(logical(1-Omega)) = mean(original_image(Omega));
    
    errList = zeros(iter, 1);
    recon_mse= zeros(iter, 1);
    normT = norm(original_image(:));
    %L = errList;

    beta = 1e-5;
    %  beta = 1e-4;

     max_beta = 10;

    optimal=zeros(dim);
    W1 = zeros(size(original_image));
    tic

    for k = 1 : iter
        Xlast = X_S;
        beta = beta * 1.05;

        % update Y 
        [Y] = prox_TNN(X_S-W1/beta,1/beta);

      
    %      X_S = (W1 + beta*Y) / (ndims(original_image)*beta);
         X_S = (W1 + beta*Y) / beta;
         X_S(Omega) = original_image(Omega); 

          % update W
             W1 = W1 + beta * (Y - X_S); 

             rho_w = 1.05;
             beta = min(rho_w * beta, max_beta);
    %     X_S = (Msum + beta*Ysum) / (ndims(original_image)*beta);
    %     X_S(Omega) = original_image(Omega);
        if mod(k, 10) == 0
            fprintf('iters= %d,x_s_psnr = %f,mse = %f\n',k,psnr(uint8(original_image),uint8(X_S)),sum(mse(original_image,X_S)));
        end

        if k>50
        % Highpass filter test image
        npd = 16;
        fltlmbd = 10;  
        [ll, lh] = lowpass(X_S, fltlmbd, npd);

        % Compute representation
        SCOpts = [];
        SCOpts.Verbose = 0;
        SCOpts.MaxMainIter = 300;
        SCOpts.rho =rho;
        SCOpts.AuxVarObj = 0;
        SCOpts.HighMemSolve = 1;

        [X, optinf2] = cbpdn_gpu(D, lh, lambda, SCOpts); 
        DX = ifft2(sum(bsxfun(@times, fft2(D, size(X,1), size(X,2)), fft2(X)),3), ...
            'symmetric');
        DX = reshape(DX,size(original_image));

        reconstruction = double(DX+ll);%dx=update gaotong
        reconstruction(reconstruction<0) = 0;
        reconstruction(Omega) = original_image(Omega);

        recon_mse(k)=sum(mse(original_image,reconstruction));
        errList(k) = norm(reconstruction(:)-Xlast(:)) / normT;
        if recon_mse(k)<sum(mse(original_image,X_S))
            X_S = reconstruction;
        end
        if (k>10 && sum(mse(original_image,optimal))>recon_mse(k))
                    optimal = reconstruction;
        end


        if (k>150 &&  recon_mse(k)> recon_mse(k-1)&&  recon_mse(k)>  recon_mse(k-2)&&  recon_mse(k)>recon_mse(k-3)&&  recon_mse(k)>recon_mse(k-4))
            errList = errList(1:k);
            break;
        end

        end
        if k<=50
            errList(k) = norm(X_S(:)-Xlast(:)) / normT;
        end
        reconstruction = optimal;
        if mod(k, 10) == 0
            fprintf('recon_psnr = %f，recon_ssim = %f,recon_mse=%f，error=%f\n',psnr(uint8(original_image),uint8(reconstruction)),ssim(uint8(original_image),uint8(reconstruction)),sum(mse(original_image,reconstruction)),errList(k)); end

    end

    toc
    fclose('all');
    beta = 1e-6;
    [XX, errListX] = HaLRTC(...
        original_image,...                      % a tensor whose elements in Omega are used for estimating missing value
        Omega,...           % the index set indicating the obeserved elements
        alpha, ...             % the coefficient of the objective function,  i.e., \|X\|_* := \alpha_i \|X_{i(i)}\|_* 
        beta,...                % the relaxation parameter. The larger, the closer to the original problem. See the function for definitions.
        500,...         % the maximum iterations
        epsilon...            % the tolerance of the relative difference of outputs of two neighbor iterations 
        );



    fprintf('PSNR: SiLRTC = %f   HaLRTC-CSC=%f\n\n', psnr(uint8(original_image), uint8(XX)), psnr(uint8(original_image), uint8(reconstruction)));

    % inwirt picture
    filename = ['/home/ltc/CSC/CSCtt_result/img/',imageName,'_mr_',num2str(mr),'_lambda_',num2str(lambda),'_tau_',num2str(tau),'_filterlambda_',num2str(filter_lambda),'_',num2str(psnr(uint8(original_image),uint8(reconstruction))),'.png'];
    imwrite(uint8(reconstruction),filename);
    imshow(uint8(reconstruction));
    % imshow(uint8(reconstruction),'border','tight','initialmagnification','fit');

end
disp('End');
