clear all; close all; clc;

rng(2019)
% t = 2*pi*sort(sqrt(rand(10000,1)));
% t = 2*pi*sort(rand(10000,1));
% X = [3*cos(t) sin(t)];

t = 2*pi*sort(.9*rand(20000,1));
X = [cos(t) sin(t)];
X = X + .01*randn(size(X));

numIter = 7;
centers = cell(numIter,1);
for k = 1:numIter
    numClust = 2*2^k;
    [L,C] = kmeans_plusplus(X',numClust);
    C = C';
    [~,ind] = sort(angle(-C(:,1) + 1i*C(:,2)));
    centers{k} = C(ind,:);
    disp(k);
end
% centers{numIter+1} = X;

figure
scatter(X(:,1),X(:,2),20,t,'filled')
hold on
scatter(centers{numIter-1}(:,1),centers{numIter-1}(:,2),50,'k','filled')
axis image


%% K_k is always a bistochastic kernel

difft = 500;

nneighbors = 15;

kernels = cell(numIter-1,1);
dist_mat = cell(numIter-1,1); %just for exploratory purposes
for k=1:numIter-1
    scale_k = centers{k};
    scale_kplus1 = centers{k+1};
    
    dist = pdist2(scale_k,scale_kplus1);
    val = sort(dist,2);
    
%     sigma = .25*median(dist(:));
    sigma = .25*mean(val(:,min(floor(size(centers{k+1},1)/2),nneighbors)));
    kernels{k}.sigma = sigma;

    A = exp(-dist.^2/sigma.^2);
    
    A = bsxfun(@times,A,1./sum(A,2));

    if k==1
        kernels{1}.normalize = 1./sqrt(mean(A,2))';
    end
    kernels{k+1}.normalize = 1./sqrt(mean(A,1));

    A = bsxfun(@times,A,kernels{k+1}.normalize/sqrt(size(A,1)));
          
    kernels{k}.K = A*A';
    
    dist_mat{k} = squareform(pdist(kernels{k}.K^difft)*sqrt(size(A,1)));
    
    kernels{k}.hierarchical = A;

    disp(k);
end
figure
subplot(1,2,1);
imagesc(dist_mat{numIter-1})
title('Diffusion Distances')
colorbar
axis image
subplot(1,2,2);
imagesc(squareform(pdist(centers{numIter-1})))    
title('Euclidean Distances')
colorbar
axis image



%% Hierarchical

difft = 500;
threshold = 1e-2;

fulldata = cell(numIter-1,1);
for level=1:numIter-1

% could use earlier levels to determine which centers needed for this
% distance matrix and estimating sigma
if level==1
    dist = pdist2(X,centers{level});
    A = exp(-dist.^2/kernels{level}.sigma.^2);
    
    fulldata{level}.A = sparse(A>threshold | bsxfun(@ge,A,max(A,[],2)-eps));
else
    A = fulldata{level-1}.A*kernels{level-1}.hierarchical;
    nearbycenters =  sparse(A>threshold);
    nearbycenters(sum(nearbycenters,2)==0,:)=ones(sum(sum(nearbycenters,2)==0),...
        size(centers{level},1));

    nnzeros = sum(sum(nearbycenters));
    I = zeros(nnzeros,1);
    J = zeros(nnzeros,1);
    K = zeros(nnzeros,1);
    ind = 0;
    for i=1:size(X,1)
        nnzeros_i = sum(nearbycenters(i,:));
        I(ind+1:ind+nnzeros_i) = i*ones(nnzeros_i,1);
        J(ind+1:ind+nnzeros_i) = find(nearbycenters(i,:)>0)';
        K(ind+1:ind+nnzeros_i) = pdist2(X(i,:),centers{level}(nearbycenters(i,:)>0,:))';
        ind = ind+nnzeros_i;
    end
%     dist = sparse(I,J,K,size(X,1),size(centers{level},1));
    A = sparse(I,J,exp(-K.^2/kernels{level}.sigma.^2),size(X,1),size(centers{level},1));
    
    fulldata{level}.A = A;%>threshold;
end


A = bsxfun(@times,A,1./sum(A,2));  % can compute per data point streaming

% reason use stored row sum here is because otherwise need entire data set
% to normalize appropriately
A = bsxfun(@times,A,kernels{level}.normalize/sqrt(size(A,1)));  

disp(level);

end

if size(A,1)<2500
    ind = 1:size(A,1);
else
    ind = sort(randperm(size(A,1),2500));
end

figure
subplot(1,2,1);
% can choose diffusion time to look at more gloal distances
imagesc(squareform(pdist(A(ind,:)*kernels{level}.K^difft))*sqrt(length(ind)))
title('Diffusion Distances')
colorbar
axis image
subplot(1,2,2);
imagesc(squareform(pdist(X(ind,:))))
title('Euclidean Distances')
colorbar
axis image

%% Ignore after here, just random notes and other curiousities

%% all data

difft = 100;
level = numIter-1; %can't do last level because don't have diffusion matrix for it

% could use earlier levels to determine which centers needed for this
% distance matrix and estimating sigma
dist = pdist2(X,centers{level});
% sigma = .25*median(dist(:));
sigma = kernels{level}.sigma;

A = exp(-dist.^2/sigma.^2);
A = bsxfun(@times,A,1./sum(A,2));  % can compute per data point streaming

% reason use stored row sum here is because otherwise need entire data set
% to normalize appropriately
A = bsxfun(@times,A,kernels{level}.normalize/sqrt(size(A,1)));  

% can choose diffusion time to look at more gloal distances
A = A*kernels{level}.K^difft;

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
