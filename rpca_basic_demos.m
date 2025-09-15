clear; close all; clc;
%% Common solver options
opts = struct('tol',1e-7,'max_iter',1000,'verbose',1);

%% ------------------------------------------------------------
% 1) Synthetic low-rank + sparse
rng(7);
m = 10; n = 10; r = 5;  rho = 0.10;   % 10% sparse corruption

A  = randn(m,r); B = randn(n,r);
L0 = A*B';                               % low-rank ground truth
Omega = rand(m,n) < rho;                 % sparse support
S0 = zeros(m,n);
S0(Omega) = 10*randn(nnz(Omega),1);      % large-magnitude sparse errors
M  = L0 + S0;

fprintf('\n=== 1) Synthetic matrix (m=%d,n=%d,r=%d, rho=%.2f) ===\n', m,n,r,rho);
tic; [L,S,out] = rpca_pcp(M, 1/sqrt(max(m,n)), opts); t1 = toc;
relL = norm(L-L0,'fro')/max(1,norm(L0,'fro'));
relS = norm(S-S0,'fro')/max(1,norm(S0,'fro'));
fprintf('Time: %.2fs | rel err L: %.3e | rel err S: %.3e\n', t1, relL, relS);

figure('Name','1)Synthetic','Color','w');
subplot(2,3,1); imagesc(L0); axis image off; title('L_0 (true low-rank)'); colormap gray;
subplot(2,3,2); imagesc(S0); axis image off; title('S_0 (true sparse)');
subplot(2,3,3); imagesc(M);  axis image off; title('M = L_0 + S_0');
subplot(2,3,4); imagesc(L);  axis image off; title(sprintf('Recovered L (rank=%d)', out.rankL));
subplot(2,3,5); imagesc(S);  axis image off; title(sprintf('Recovered S (Sparsity=%d)', out.nnzS));
subplot(2,3,6); imagesc(L+S); axis image off; title('Final: M = L + S');


%% ------------------------------------------------------------
%  2) images (lighting + black box)
fprintf('\n=== Image stack with lighting + black boxes (Cameraman) ===\n');
I = im2double(imresize(imread('cameraman.tif'), [128 128])); 
K = 24;                                      % number of images
stack = zeros(size(I,1), size(I,2), K);

% Create K variants: lighting change + gradient + some occlusions
[Xg,Yg] = meshgrid(linspace(-1,1,size(I,2)), linspace(-1,1,size(I,1)));
for k = 1:K
    scale = 0.6 + 0.8 * (k-1)/(K-1);         % lighting scale 0.6 -> 1.4
    grad  = 0.15 * (0.5*Xg - 0.3*Yg);        % shading
    Ik = max(min(scale*I + grad,1),0);
    if mod(k,4)==0
        % add a black box in varying location
        oc = false(size(I)); 
        r0 = 20+3*k; c0 = 30+2*k; h=30; w=40;
        r1 = min(size(I,1), r0+h); c1 = min(size(I,2), c0+w);
        oc(r0:r1, c0:c1) = true;
        Ik(oc) = 0;                           
    end
    stack(:,:,k) = Ik;
end

M3 = reshape(stack, [], K);
tic; [L3,S3,out3] = rpca_pcp(M3, 1/sqrt(max(numel(I),K)), opts); t3 = toc;
L3_stack = reshape(L3, size(I,1), size(I,2), K);
S3_stack = reshape(S3, size(I,1), size(I,2), K);
fprintf('Time: %.2fs | rank(L)=%d | relres=%.2e\n', t3, out3.rankL, out3.relres);

% Show a few columns before/after
pick = [4 8 12 16 20 24];
figure('Name','2) Images','Color','w');
for i=1:numel(pick)
    t = pick(i);
    subplot(3,numel(pick), i);    imagesc(stack(:,:,t)); axis image off; title(sprintf('Orig %d',t));
    subplot(3,numel(pick), i+numel(pick)); imagesc(L3_stack(:,:,t)); axis image off; title('Low-rank');
    subplot(3,numel(pick), i+2*numel(pick)); imagesc(abs(S3_stack(:,:,t))); axis image off; title('Sparse');
end
colormap gray; sgtitle('Cameraman: Original / Low-rank / Sparse');

