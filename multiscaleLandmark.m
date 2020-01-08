clear all; close all; clc;

rng(2019)
% t = 2*pi*sort(sqrt(rand(10000,1)));
% t = 2*pi*sort(rand(10000,1));
% X = [3*cos(t) sin(t)];

% generate 2D data
t = 2*pi*sort(.9*rand(2000,1));
X = [cos(t) sin(t)];
X = X + .05*randn(size(X));

numIter = 6;
centers = cell(numIter,1);
for k = 1:numIter
    numClust = 2^k;
    [L,C] = kmeans_plusplus(X',numClust);
    C = C';
    [~,ind] = sort(angle(-C(:,1) + 1i*C(:,2)));
    centers{k} = C(ind,:);
    disp(k);
end
% Centers contains kmeans++ samples of size 2,4,8,..,2^numIter and then X
% itself
centers{numIter+1} = X;

figure
%Plot all data with originating angle as color.
scatter(X(:,1),X(:,2),20,t,'filled')
hold on
%plot centers for largest kmeans++
scatter(centers{numIter}(:,1),centers{numIter}(:,2),50,'k','filled')
axis image


%% K_k is always a bistochastic kernel

difft = 5;

kernel_normalize = cell(numIter,1);
kernels = cell(numIter-1,1);
dist_mat = cell(numIter-1,1); %just for exploratory purposes
iters=3
for k=1:iters
    scale_k = centers{k};
    scale_kplus1 = centers{k+1};
    
    dist = pdist2(scale_k,scale_kplus1); %compute pairwise distances
    sigma = .25*median(dist(:));
    A = exp(-dist.^2/sigma.^2);
    
    A = bsxfun(@times,A,1./sum(A,2)); % normalize each column of A to 1

    kernel_normalize{k+1} = 1./sqrt(mean(A,1));
    A = bsxfun(@times,A,kernel_normalize{k+1}/sqrt(size(A,1)));
          
    kernels{k} = A*A';
    
    dist_mat{k} = squareform(pdist(kernels{k}^difft)*sqrt(size(A,1)));
    
    disp(k)
    disp(size(dist_mat{k}));
end
figure
subplot(1,2,1);
imagesc(dist_mat{iters})
title('Diffusion Distances')
colorbar
axis image
subplot(1,2,2);
imagesc(squareform(pdist(centers{numIter-1})))    
title('Euclidean Distances')
colorbar
axis image

%% all data

difft = 5;
level = numIter-1; %can't do last level because don't have diffusion matrix for it

% could use earlier levels to determine which centers needed for this
% distance matrix and estimating sigma
dist = pdist2(X,centers{level});
sigma = .25*median(dist(:));

A = exp(-dist.^2/sigma.^2);
A = bsxfun(@times,A,1./sum(A,2));  % can compute per data point streaming

% reason use stored row sum here is because otherwise need entire data set
% to normalize appropriately
A = bsxfun(@times,A,kernel_normalize{level}/sqrt(size(A,1)));  

% can choose diffusion time to look at more gloal distances
A = A*kernels{level}^difft;

if size(A,1)<2500
    figure
    subplot(1,2,1);
    imagesc(squareform(pdist(A))*sqrt(size(A,1)))
    title('Diffusion Distances')
    colorbar
    axis image
    subplot(1,2,2);
    imagesc(squareform(pdist(X)))
    title('Euclidean Distances')
    colorbar
    axis image
end

%% Ignore after here, just random notes and other curiousities

%% other direction, also works (so k is centers, k+1 is extended centers)
difft = 5;

kernels_asym = cell(numIter,1);
kernels_sym = cell(numIter,1);
dist_mat = cell(numIter,1);
for k=2:numIter
    scale_k = centers{k-1};
    scale_kplus1 = centers{k};
    
    dist = pdist2(scale_kplus1,scale_k);
    sigma = .25*median(dist(:));
    A = exp(-dist.^2/sigma.^2);
    A = bsxfun(@times,A,1./sum(A,2));
    A = bsxfun(@times,A,1./sqrt(sum(A,1)));
%     K_k = A'*bsxfun(@times,A,1./sum(A,1));
    K_k = A*A';
    kernels_asym{k} = A;
    kernels_sym{k} = K_k;
    
    dist_mat{k} = squareform(pdist(K_k^difft));

    disp(k);
end
figure
subplot(1,2,1);
imagesc(dist_mat{numIter})
colorbar
axis image
subplot(1,2,2);
imagesc(squareform(pdist(centers{numIter})))
colorbar
axis image


%% Doesn't work well, but only way I can think of to connect different levels
kernels_multi = cell(numIter-1,1);
kernels_multi{numIter-1} = kernels_sym{numIter-1};
for k=numIter-2:-1:1
    K = kernels_asym{k}'*kernels_multi{k+1}*kernels_asym{k};
    kernels_multi{k} = K;
end

for k=1:numIter-1
    subplot(numIter-1,2,2*k-1);
    imagesc(kernels_sym{k}^2)
    axis image
    subplot(numIter-1,2,2*k);
    imagesc(kernels_multi{k})
    axis image
end
