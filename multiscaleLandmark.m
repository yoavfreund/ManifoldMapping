clear all; close all; clc;


%% Creating Simple Data, can be ignored for general algorithm
% data is n x d matrix X

%1D curve with gap
% N = 50000;
% d = 2;
% t = 2*pi*sort(.9*rand(N,1));
% X = [cos(t) sin(t)];
% X = X + .03*randn(size(X));
% scatter(X(:,1),X(:,2),20)

%high dim spheres with 2D plane connecting 
d=5;
N1 = 5000; m1=d; %change to make spheres at ends higher dimensional
N2 = 5000; m2=d; %change to make spheres at ends higher dimensional
distBetween = 2;
Nmiddle = 2000; mMiddle = 2;
thickMiddle = 1;

x1 = randn(N1,m1);
x1 = bsxfun(@times,x1,1./sqrt(sum(x1.^2,2)));
[~,ind] = sort(x1(:,1));
x1 = x1(ind,:);

x2 = randn(N2,m2);
x2 = bsxfun(@times,x2,1./sqrt(sum(x2.^2,2)));
x2 = bsxfun(@plus,x2,[distBetween+2,zeros(1,m2-1)]);
x2 = [x2 zeros(N2,m1-m2)];
[~,ind] = sort(x2(:,1));
x2 = x2(ind,:);

x3 = rand(Nmiddle,mMiddle);
[~,ind] = sort(x3(:,1));
x3 = x3(ind,:);
x3 = bsxfun(@times,x3,[2+distBetween,thickMiddle*ones(1,mMiddle-1)]);
x3 = bsxfun(@minus,x3,[0,thickMiddle/2*ones(1,mMiddle-1)]);
index1 = sqrt(sum(x3.^2,2))<=1;
index2 = sqrt(sum( bsxfun(@minus,x3,[2+distBetween,zeros(1,mMiddle-1)]).^2,2))<=1;
x3(index1 | index2,:) = [];
x3 = [x3 zeros(size(x3,1),m1 - mMiddle)];
[~,ind] = sort(x3(:,1));
x3 = x3(ind,:);

X = [x1;x2;x3];
scatter3(X(:,1),X(:,2),X(:,3),20);
axis image



%% Run kmeans++ with 2^k number of centers
% currently run in very foolish fashion.  This is heart of what can be
% adjusted to work in streaming

% returns centers, which is a cell containing the centers from kmeans++ at
% each level
% centers{k} = C_k has 2^k points in it.

numIter = 9;
centers = cell(numIter,1);
for k = 1:numIter
    numClust = 2^k;
    [L,C] = kmeans_plusplus(X',numClust);
    C = C';
    
    % just sorting for visualization purposes
    if d==2
        [~,ind] = sort(angle(-C(:,1) + 1i*C(:,2)));
        centers{k} = C(ind,:);
    else
        [~,ind] = sort(C(:,1));
        centers{k} = C(ind,:);
    end
    disp(k);
end
% centers{numIter+1} = X;

figure
if d==2
    scatter(X(:,1),X(:,2),20,t,'filled')
    hold on
    scatter(centers{numIter-1}(:,1),centers{numIter-1}(:,2),50,'k','filled')
    axis image
else 
    scatter3(X(:,1),X(:,2),X(:,3),20,'filled')
    hold on
    scatter3(centers{numIter-1}(:,1),centers{numIter-1}(:,2), ...
        centers{numIter-1}(:,3),50,'k','filled')
    axis image
end


%% Computing affinity between C_k using C_k+1
% returns kernels, which stores relationships between points in 

%parameter for choosing bandwidth of Gaussian kernel
% should be some power of intrinsic dimension.  8 coefficient is
% arbitraray.
if d==2
    knnMax = 8*2^1;
else
    % choose intrinsic dimension 2 here because we aren't going to be able
    % to model the spheres at ends correctly due to large dimension
    knnMax = 8*2^2; 
end

kernels = cell(numIter-1,1);
for k=1:numIter-1
    C_k = centers{k};
    C_kplus1 = centers{k+1};
    
    % distance from points in C_k to points in C_k+1
    dist = pdist2(C_k,C_kplus1);
    
    % choose bandwidth to be average to knnMax nearest neighbor in C_k+1
    % nneighbors has min for when #|C_k+1| is very small.
    nneighbors = min(floor(size(centers{k+1},1)/2),knnMax); 
    val = sort(dist,2);
    sigma = .25*mean(val(:,nneighbors));
    kernels{k}.sigma = sigma; %store bandwidth for later calculations on full data

    % compute Gaussian kernel on C_k x C_k+1 at appropriate bandwidth
    A = exp(-dist.^2/sigma.^2);
    
    % row normalize A so its a transition matrix for a walk starting at
    % points in C_k and going to points in C_k+1
    A = bsxfun(@times,A,1./sum(A,2));

    % estimate sqareroot of density of points in C_k+1
    % stored if want to use later on full data
    kernels{k+1}.normalize = 1./sqrt(mean(A,1));
    normalization = kernels{k+1}.normalize/sqrt(size(A,1)); 

    % column normalize A
    A = bsxfun(@times,A,normalization);
          
    % build affinity between C_k and C_k using diffusion through points in
    % C_k+1.  Is of use to speed up computation of diffusion times on full
    % data.
    kernels{k}.K = A*A';
        
    % store A, which records which points in C_k+1 are close to points in
    % C_k.  Is used later to reduce computational complexity
    kernels{k}.hierarchical = A;

    disp(k);
end




%% Computing affinity between all data and C_k hierarchically and quickly
% returns A, an N x #|C_k| sparse matrix of affinity to neighbors

% diffusion time for computting diffusion distances at the end
difft = 250;

% cutoff for what's considered too small of affinity.  Anything below
% threshold is zeroed out.
threshold = 1e-2;

fulldata = cell(numIter-1,1);
for level=1:numIter-1

    % if on first level, need to measure distance to all centers 
    if level==1
        dist = pdist2(X,centers{level});
        
        % compute affinity using predetermined bandwidth
        A = exp(-dist.^2/kernels{level}.sigma.^2);

        % indicator of which centers each point in full data is neighboring.  
        % Ensures each point has at least one neighbor.
        % Neighbor if either:
        % - affinity is greater than threshold
        % - affinity is largest is row
        fulldata{level}.A = sparse(A>threshold | bsxfun(@ge,A,max(A,[],2)-eps));
    else
        % for full data, find neighbors in C_k+1 using knowledge of neighbors in C_k and
        % neighborhoods of C_k in C_k+1
        A = fulldata{level-1}.A*kernels{level-1}.hierarchical;
        
        % turn into indicator function of neighbors by checking against threshold
        nearbycenters =  sparse(A>threshold);
        
        % if any point is assigned no neighbors in C_k+1, then set to
        % measure to all points in C_k+1 since uncertain of location
        nearbycenters(sum(nearbycenters,2)==0,:)=ones(sum(sum(nearbycenters,2)==0),...
            size(centers{level},1));
        

        % for each point in data, measure distance to it's relevant
        % neighbors in C_k+1 only.  Not all of C_k+1
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
        
        % compute affinity matrix in sparse format
        A = sparse(I,J,exp(-K.^2/kernels{level}.sigma.^2),size(X,1),size(centers{level},1));

        
        
        % store for next level of centers
        fulldata{level}.A = sparse(A>threshold);
    end  

    disp(level);

end

% Row normalize A for all points in data.  Can compute per data point,
% so it's condusive to streaming
A = bsxfun(@times,A,1./sum(A,2));  

% Choose a few points from full data to look at diffusion distance matrix
if size(A,1)<2500
    ind = 1:size(A,1);
else
    ind = sort(randperm(size(A,1),2500));
end
% New featurs of full data stored in matrix A are very sparse.  Can use
% diffusion time on affinity between C_k and C_k to get larger diffusion
% neighborhoods
diffusedData = A(ind,:)* (kernels{level}.K^difft);


figure
subplot(1,2,1);
imagesc(squareform(pdist(diffusedData)))
title('Diffusion Distances')
colorbar
axis image
subplot(1,2,2);

% comparision to Euclidean distance calculation
imagesc(squareform(pdist(X(ind,:))))
title('Euclidean Distances')
colorbar
axis image
