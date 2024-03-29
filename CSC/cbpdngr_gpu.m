function [Y, optinf] = cbpdngr_gpu(D, S, lambda, mu, opt)

% cbpdngr -- Convolutional Basis Pursuit DeNoising with Gradient Regularization
%
%         argmin_{x_k} (1/2)||\sum_k d_k * x_k - s||_2^2 +
%                           lambda \sum_k ||x_k||_1 +
%                           (mu/2) \sum_k ||G_r d_k * x_k||_2^2 +
%                           (mu/2) \sum_k ||G_c d_k * x_k||_2^2
%
%         The solution is computed using an ADMM approach (see
%         boyd-2010-distributed) with efficient solution of the main
%         linear systems (see wohlberg-2016-efficient and
%         wohlberg-2016-convolutional2).
%
% Usage:
%       [Y, optinf] = cbpdngr(D, S, lambda, mu, opt);
%
% Input:
%       D           Dictionary filter set (3D array)
%       S           Input image
%       lambda      Regularization parameter (l1)
%       mu          Regularization parameter (l2 of gradient)
%       opt         Algorithm parameters structure
%
% Output:
%       Y           Dictionary coefficient map set (3D array)
%       optinf      Details of optimisation
%
%
% Options structure fields:
%   Verbose          Flag determining whether iteration status is displayed.
%                    Fields are iteration number, functional value,
%                    data fidelity term, l1 regularisation term, gradient
%                    regularisation term, and primal and dual residuals
%                    (see Sec. 3.3 of boyd-2010-distributed). The value of
%                    rho is also displayed if options request that it is
%                    automatically adjusted.
%   MaxMainIter      Maximum main iterations
%   AbsStopTol       Absolute convergence tolerance (see Sec. 3.3.1 of
%                    boyd-2010-distributed)
%   RelStopTol       Relative convergence tolerance (see Sec. 3.3.1 of
%                    boyd-2010-distributed)
%   L1Weight         Weighting array for coefficients in l1 norm of X
%   GrdWeight        Weighting array for coefficients in l2 norm of
%                    gradient of X
%   Y0               Initial value for Y
%   U0               Initial value for U
%   rho              ADMM penalty parameter
%   AutoRho          Flag determining whether rho is automatically updated
%                    (see Sec. 3.4.1 of boyd-2010-distributed)
%   AutoRhoPeriod    Iteration period on which rho is updated
%   RhoRsdlRatio     Primal/dual residual ratio in rho update test
%   RhoScaling       Multiplier applied to rho when updated
%   AutoRhoScaling   Flag determining whether RhoScaling value is
%                    adaptively determined (see wohlberg-2015-adaptive). If
%                    enabled, RhoScaling specifies a maximum allowed
%                    multiplier instead of a fixed multiplier.
%   RhoRsdlTarget    Residual ratio targeted by auto rho update policy.
%   StdResiduals     Flag determining whether standard residual definitions
%                    (see Sec 3.3 of boyd-2010-distributed) are used instead
%                    of normalised residuals (see wohlberg-2015-adaptive)
%   RelaxParam       Relaxation parameter (see Sec. 3.4.3 of
%                    boyd-2010-distributed)
%   NonNegCoef       Flag indicating whether solution should be forced to
%                    be non-negative
%   NoBndryCross     Flag indicating whether all solution coefficients
%                    corresponding to filters crossing the image boundary
%                    should be forced to zero.
%   AuxVarObj        Flag determining whether objective function is computed
%                    using the auxiliary (split) variable
%   HighMemSolve     Use more memory for a slightly faster solution
%
%
% Author: Brendt Wohlberg <brendt@lanl.gov>  Modified: 2016-07-01
%
% This file is part of the SPORCO library. Details of the copyright
% and user license can be found in the 'License' file distributed with
% the library.

gS = gpuArray(S);
gD = gpuArray(D);
glambda = gpuArray(lambda);
gmu = gpuArray(mu);

if nargin < 5,
  opt = [];
end
checkopt(opt, defaultopts([]));
opt = defaultopts(opt);

% Set up status display for verbose operation
hstr = 'Itn   Fnc       DFid      l1        Grd        r         s      ';
sfms = '%4d %9.2e %9.2e %9.2e %9.2e %9.2e %9.2e';
nsep = 64;
if opt.AutoRho,
  hstr = [hstr '   rho  '];
  sfms = [sfms ' %9.2e'];
  nsep = nsep + 10;
end
if opt.Verbose && opt.MaxMainIter > 0,
  disp(hstr);
  disp(char('-' * ones(1,nsep)));
end

% Start timer
tstart = tic;

% Collapsing of trailing singleton dimensions greatly complicates
% handling of both SMV and MMV cases. The simplest approach would be
% if S could always be reshaped to 4d, with dimensions consisting of
% image rows, image cols, a single dimensional placeholder for number
% of filters, and number of measurements, but in the single
% measurement case the third dimension is collapsed so that the array
% is only 3d.
if size(S,3) > 1,
  xsz = [size(S,1) size(S,2) size(gD,3) size(S,3)];
  hrm = [1 1 1 size(S,3)];
  % Insert singleton 3rd dimension (for number of filters) so that
  % 4th dimension is number of images in input s volume
  gS = reshape(gS, [size(S,1) size(S,2) 1 size(S,3)]);
else
  xsz = [size(S,1) size(S,2) size(gD,3) 1];
  hrm = 1;
end
xrm = [1 1 size(gD,3)];

% Compute filters in DFT domain
gDf = fft2(gD, size(S,1), size(S,2));
grv = gpuArray([-1 1]);
gGrf = fft2(grv, size(S,1), size(S,2));
gcv = gpuArray([-1 1]');
gGcf = fft2(gcv, size(S,1), size(S,2));
if isscalar(opt.GrdWeight),
  opt.GrdWeight = opt.GrdWeight * ones(size(gD,3), 1);
end
wgr = reshape(opt.GrdWeight, [1 1 length(opt.GrdWeight)]);
gGfW = bsxfun(@times, conj(gGrf).*gGrf + conj(gGcf).*gGcf, wgr);

%all you add is one line 
% gDoDT = bsxfun(@times,conj(gDf),gDf);
% gGfW = bsxfun(@times,gDoDT,gGfW);

% Convolve-sum and its Hermitian transpose
gDop = @(x) sum(bsxfun(@times, gDf, x), 3);
gDHop = @(x) bsxfun(@times, conj(gDf), x);
% Compute signal in DFT domain
gSf = fft2(gS);
% S convolved with all filters in DFT domain
gDSf = gDHop(gSf);


% Default lambda is 1/10 times the lambda value beyond which the
% solution is a zero vector
if nargin < 3 | isempty(lambda),
  gb = ifft2(DHop(gSf), 'symmetric');
  glambda = 0.1*max(vec(abs(gb)));
end

% Set up algorithm parameters and initialise variables
grho = gpuArray(opt.rho);
if isempty(grho), grho = 50*lambda+1; end;
if isempty(opt.RhoRsdlTarget),
  if opt.StdResiduals,
    opt.RhoRsdlTarget = 1;
  else
    opt.RhoRsdlTarget = 1 + (18.3).^(log10(gather(glambda)) + 1);
  end
end
if opt.HighMemSolve,
  gcn = bsxfun(@rdivide, gDf, gmu*gGfW + grho);
  gcd = sum(gDf.*bsxfun(@rdivide, conj(gDf), gmu*gGfW + grho), 3) + 1.05;
  gC = bsxfun(@rdivide, gcn, gcd);
  clear cn cd;
else
  gC = [];
end
gNx = prod(gpuArray(xsz));
optinf = struct('itstat', [], 'opt', opt);
gr = gpuArray(Inf);
gs = gpuArray(Inf);
gepri = gpuArray(0);
gedua = gpuArray(0);

% Initialise main working variables
% X = [];
if isempty(opt.Y0),
  gY = gpuArray.zeros(xsz, class(S));
else
  gY = gpuArray(opt.Y0);
end
gYprv = gY;
if isempty(opt.U0),
  if isempty(opt.Y0),
    gU = gpuArray.zeros(xsz, class(S));
  else
    gU = (glambda/grho)*sign(gY);
  end
else
  gU = gpuArray(opt.U0);
end

% Main loop
k = 1;
while k <= opt.MaxMainIter && (gr > gepri | gs > gedua),

  % Solve X subproblem
  gXf = solvedbd_sm(gDf, gmu*gGfW + grho, gDSf + grho*fft2(gY - gU), gC);
  gX = ifft2(gXf, 'symmetric');

  % See pg. 21 of boyd-2010-distributed
  if opt.RelaxParam == 1,
    gXr = gX;
  else
    gXr = opt.RelaxParam*gX + (1-opt.RelaxParam)*gY;
  end

  % Solve Y subproblem
  gY = shrink(gXr + gU, (glambda/grho)*opt.L1Weight);
  if opt.NonNegCoef,
    gY(gY < 0) = 0;
  end
  if opt.NoBndryCross,
    gY((end-size(gD,1)+2):end,:,:,:) = 0;
    gY(:,(end-size(gD,2)+2):end,:,:) = 0;
  end

  % Update dual variable
  gU = gU + gXr - gY;

  % Compute data fidelity term in Fourier domain (note normalisation)
  if opt.AuxVarObj,
    gYf = fft2(gY); % This represents unnecessary computational cost
    gJdf = sum(vec(abs(sum(bsxfun(@times,gDf,gYf),3)-gSf).^2))/(2*xsz(1)*xsz(2));
    gJl1 = sum(abs(vec(bsxfun(@times, opt.L1Weight, gY))));
    gJgr = sum(vec((bsxfun(@times, gGfW, conj(gYf).*gYf))))/(2*xsz(1)*xsz(2));
  else
    gJdf = sum(vec(abs(sum(bsxfun(@times,gDf,gXf),3)-gSf).^2))/(2*xsz(1)*xsz(2));
    gJl1 = sum(abs(vec(bsxfun(@times, opt.L1Weight, gX))));
    gJgr = sum(vec((bsxfun(@times, gGfW, conj(gXf).*gXf))))/(2*xsz(1)*xsz(2));
  end
  gJfn = gJdf + glambda*gJl1 + gmu*gJgr;

  gnX = norm(gX(:)); gnY = norm(gY(:)); gnU = norm(gU(:));
  if opt.StdResiduals,
    % See pp. 19-20 of boyd-2010-distributed
    gr = norm(vec(gX - gY));
    gs = norm(vec(grho*(gYprv - gY)));
    gepri = sqrt(gNx)*opt.AbsStopTol+max(gnX,gnY)*opt.RelStopTol;
    gedua = sqrt(gNx)*opt.AbsStopTol+grho*gnU*opt.RelStopTol;
  else
    % See wohlberg-2015-adaptive
    gr = norm(vec(gX - gY))/max(gnX,gnY);
    gs = norm(vec(gYprv - gY))/gnU;
    gepri = sqrt(gNx)*opt.AbsStopTol/max(gnX,gnY)+opt.RelStopTol;
    gedua = sqrt(gNx)*opt.AbsStopTol/(grho*gnU)+opt.RelStopTol;
  end

  % Record and display iteration details
  tk = toc(tstart);
  optinf.itstat = [optinf.itstat; [k gather(gJfn) gather(gJdf) gather(gJl1) gather(gJgr) gather(gr) gather(gs) gather(gepri) gather(gedua) grho tk]];
  if opt.Verbose,
    if opt.AutoRho,
      disp(sprintf(sfms, k, Jfn, Jdf, Jl1, Jgr, r, s, rho));
    else
      disp(sprintf(sfms, k, Jfn, Jdf, Jl1, Jgr, r, s));
    end
  end

  % See wohlberg-2015-adaptive and pp. 20-21 of boyd-2010-distributed
  if opt.AutoRho,
    if k ~= 1 && mod(k, opt.AutoRhoPeriod) == 0,
      if opt.AutoRhoScaling,
        grhomlt = sqrt(gr/(gs*opt.RhoRsdlTarget));
        if grhomlt < 1, grhomlt = 1/grhomlt; end
        if grhomlt > opt.RhoScaling, grhomlt = opt.RhoScaling; end
      else
        grhomlt = opt.RhoScaling;
      end
      grsf = 1;
      if gr > opt.RhoRsdlTarget*opt.RhoRsdlRatio*gs, grsf = grhomlt; end
      if gs > (opt.RhoRsdlRatio/opt.RhoRsdlTarget)*gr, grsf = 1/grhomlt; end
      grho = grsf*grho;
      gU = gU/grsf;
      if opt.HighMemSolve && grsf ~= 1,
        gcn = bsxfun(@rdivide, gDf, gmu*gGfW + grho);
        gcd = sum(gDf.*bsxfun(@rdivide, conj(gDf), gmu*gGfW + grho)*1, 3) + 1.05;
        gC = bsxfun(@rdivide, gcn, gcd);
        clear cn cd;
      end
    end
  end

  gYprv = gY;
  k = k + 1;

end

% Record run time and working variables
optinf.runtime = toc(tstart);
optinf.X = gather(gX);
optinf.Xf = gather(gXf);
optinf.Y = gather(gY);
optinf.U = gather(gU);
optinf.lambda = gather(glambda);
optinf.mu = gather(gmu);
optinf.rho = gather(grho);
Y = gather(gY);
c=gpuArray(S);
a=gX(:,:,32,:);
b=reshape(a,[256,256,3]);
% filename = ['/home/ltc/CSC/Copy_of_PWLS-CSCGR-master/CSCtt_result/tt_M_dic_new/',num2str(k),'_star',num2str(psnr(c,b)),'.png'];
% imwrite(b,filename);


% End status display for verbose operation
if opt.Verbose && opt.MaxMainIter > 0,
  disp(char('-' * ones(1,nsep)));
end

return


function u = vec(v)

  u = v(:);

return


function u = shrink(v, lambda)

  if isscalar(lambda),
    u = sign(v).*max(0, abs(v) - lambda);
  else
    u = sign(v).*max(0, bsxfun(@minus, abs(v), lambda));
  end

return


function opt = defaultopts(opt)

  if ~isfield(opt,'Verbose'),
    opt.Verbose = 0;
  end
  if ~isfield(opt,'MaxMainIter'),
    opt.MaxMainIter = 1000;
  end
  if ~isfield(opt,'AbsStopTol'),
    opt.AbsStopTol = 0;
  end
  if ~isfield(opt,'RelStopTol'),
    opt.RelStopTol = 1e-4;
  end
  if ~isfield(opt,'L1Weight'),
    opt.L1Weight = 1;
  end
  if ~isfield(opt,'GrdWeight'),
    opt.GrdWeight = 1;
  end
  if ~isfield(opt,'Y0'),
    opt.Y0 = [];
  end
  if ~isfield(opt,'U0'),
    opt.U0 = [];
  end
  if ~isfield(opt,'rho'),
    opt.rho = [];
  end
  if ~isfield(opt,'AutoRho'),
    opt.AutoRho = 1;
  end
  if ~isfield(opt,'AutoRhoPeriod'),
    opt.AutoRhoPeriod = 1;
  end
  if ~isfield(opt,'RhoRsdlRatio'),
    opt.RhoRsdlRatio = 1.2;
  end
  if ~isfield(opt,'RhoScaling'),
    opt.RhoScaling = 100;
  end
  if ~isfield(opt,'AutoRhoScaling'),
    opt.AutoRhoScaling = 1;
  end
  if ~isfield(opt,'RhoRsdlTarget'),
    opt.RhoRsdlTarget = [];
  end
  if ~isfield(opt,'StdResiduals'),
    opt.StdResiduals = 0;
  end
  if ~isfield(opt,'RelaxParam'),
    opt.RelaxParam = 1.8;
  end
  if ~isfield(opt,'NonNegCoef'),
    opt.NonNegCoef = 0;
  end
  if ~isfield(opt,'NoBndryCross'),
    opt.NoBndryCross = 0;
  end
  if ~isfield(opt,'AuxVarObj'),
    opt.AuxVarObj = 0;
  end
  if ~isfield(opt,'HighMemSolve'),
    opt.HighMemSolve = 0;
  end

return
