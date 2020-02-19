%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% This is the code for the U-SPEC algorithm, which is proposed in   %
% the following paper:                                              %
%                                                                   %
% D. Huang, C.-D. Wang, J.-S. Wu, J.-H. Lai, and C.-K. Kwoh.        %
% "Ultra-Scalable Spectral Clustering and Ensemble Clustering."     %
% IEEE Transactions on Knowledge and Data Engineering, 2020.        %
% DOI: https://doi.org/10.1109/TKDE.2019.2903410                    %
%                                                                   %
% The code has been tested in Matlab R2016a and Matlab R2016b.      %
% Website: https://www.researchgate.net/publication/330760669       %
% Written by Huang Dong. (huangdonghere@gmail.com)                  %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function labels = USPEC(fea, Ks, distance, p, Knn, maxTcutKmIters, cntTcutKmReps)

if nargin < 7
    cntTcutKmReps = 3; 
end
if nargin < 6
    maxTcutKmIters = 100; % maxTcutKmIters and cntTcutKmReps are used to limit the iterations of the k-means discretization in Tcut.
end
if nargin < 5
    Knn = 5; % The number of nearest neighbors.
end
if nargin < 4
    p = 1000; % The number of representatives.
end
if nargin < 3
    distance = 'euclidean'; % Use Euclidean distance by default. 
end

N = size(fea,1);
if p>N
    p = N;
end

warning('off');

%% Get $p$ representatives by hybrid selection
RpFea = getRepresentativesByHybridSelection(fea, p, distance);

%% Approx. KNN
% 1. partition RpFea into $cntRepCls$ rep-clusters
cntRepCls = floor(sqrt(p));
% 2. find the center of each rep-cluster
if strcmp(distance,'euclidean')
    [repClsLabel, repClsCenters] = litekmeans(RpFea,cntRepCls,'MaxIter',20);
else
    [repClsLabel, repClsCenters] = litekmeans(RpFea,cntRepCls,'MaxIter',20,'Distance',distance);
end
% 3. Pre-compute the distance between N objects and the $cntRepCls$
% rep-cluster centers
centerDist = pdist2_fast(fea, repClsCenters, distance);

%% Find the nearest rep-cluster (in RpFea) for each object
[~,minCenterIdxs] = min(centerDist,[],2); clear centerDist
cntRepCls = size(repClsCenters,1);
%% Then find the nearest representative in the nearest rep-cluster for each object.
nearestRepInRpFeaIdx = zeros(N,1); 
for i = 1:cntRepCls
    [~,nearestRepInRpFeaIdx(minCenterIdxs==i)] = min(pdist2_fast(fea(minCenterIdxs==i,:),RpFea(repClsLabel==i,:), distance),[],2);
    tmp = find(repClsLabel==i);
    nearestRepInRpFeaIdx(minCenterIdxs==i) = tmp(nearestRepInRpFeaIdx(minCenterIdxs==i));
end
clear repClsCenters repClsLabel minCenterIdxs tmp

%% For each object, compute its distance to the candidate neighborhood of its nearest representative (in RpFea)
neighSize = 10*Knn; % The candidate neighborhood size.
RpFeaW = pdist2_fast(RpFea,RpFea,distance);
[~,RpFeaKnnIdx] = sort(RpFeaW,2); clear RpFeaW
RpFeaKnnIdx = RpFeaKnnIdx(:,1:floor(neighSize+1)); % Pre-compute the candidate neighborhood for each representative.
RpFeaKnnDist = zeros(N,size(RpFeaKnnIdx,2));
for i = 1:p
    % fea(nearestRepInRpFeaIdx==i,:) denotes the objects (in fea) whose nearest representative is the i-th representative (in RpFea).
    RpFeaKnnDist(nearestRepInRpFeaIdx==i,:) = pdist2_fast(fea(nearestRepInRpFeaIdx==i,:), RpFea(RpFeaKnnIdx(i,:),:), distance);
end
clear fea RpFea
RpFeaKnnIdxFull = RpFeaKnnIdx(nearestRepInRpFeaIdx,:);

%% Get the final KNN according to the candidate neighborhood.
knnDist = zeros(N,Knn);
knnTmpIdx = knnDist;
knnIdx = knnDist;
for i = 1:Knn
    [knnDist(:,i),knnTmpIdx(:,i)] = min(RpFeaKnnDist,[],2);
    temp = (knnTmpIdx(:,i)-1)*N+[1:N]';
    RpFeaKnnDist(temp) = 1e100;    
    knnIdx(:,i) = RpFeaKnnIdxFull(temp);
end
clear RpFeaKnnIdx knnTmpIdx temp nearestRepInRpFeaIdx RpFeaKnnIdxFull RpFeaKnnDist

%% Compute the cross-affinity matrix B for the bipartite graph

if strcmp(distance,'cosine') 
    Gsdx = 1-knnDist;
else
    knnMeanDiff = mean(knnDist(:)); % use the mean distance as the kernel parameter $\sigma$
    Gsdx = exp(-(knnDist.^2)/(2*knnMeanDiff^2)); clear knnDist knnMeanDiff
end

Gsdx(Gsdx==0) = eps;
Gidx = repmat([1:N]',1,Knn);
B=sparse(Gidx(:),knnIdx(:),Gsdx(:),N,p); clear Gsdx Gidx knnIdx

labels = zeros(N, numel(Ks));
for iK = 1:numel(Ks)
    labels(:,iK) = Tcut_for_bipartite_graph(B,Ks(iK),maxTcutKmIters,cntTcutKmReps);
end

function RpFea = getRepresentativesByHybridSelection(fea, pSize, distance, cntTimes)
% Huang Dong. Mar. 20, 2019.
% Select $pSize$ representatives by hybrid selection.
% First, randomly select $pSize * cntTimes$ candidate representatives.
% Then, partition the candidates into $pSize$ clusters by k-means, and get
% the $pSize$ cluster centers as the final representatives.

if nargin < 4
    cntTimes = 10;
end

N = size(fea,1);
bigPSize = cntTimes*pSize;
if pSize>N
    pSize = N;
end
if bigPSize>N
    bigPSize = N;
end

rand('state',sum(100*clock)*rand(1));
bigRpFea = getRepresentivesByRandomSelection(fea, bigPSize);

if strcmp(distance,'euclidean')
    [~, RpFea] = litekmeans(bigRpFea,pSize,'MaxIter',10);
else
    [~, RpFea] = litekmeans(bigRpFea,pSize,'MaxIter',10,'Distance',distance);
end

function [RpFea,selectIdxs] = getRepresentivesByRandomSelection(fea, pSize)
% Huang Dong. Mar. 20, 2019.
% Randomly select pSize rows from fea.

N = size(fea,1);
if pSize>N
    pSize = N;
end
selectIdxs = randperm(N,pSize);
RpFea = fea(selectIdxs,:);

function labels = Tcut_for_bipartite_graph(B,Nseg,maxKmIters,cntReps)
% B - |X|-by-|Y|, cross-affinity-matrix

if nargin < 4
    cntReps = 3;
end
if nargin < 3
    maxKmIters = 100;
end

[Nx,Ny] = size(B);
if Ny < Nseg
    error('Need more columns!');
end

dx = sum(B,2);
dx(dx==0) = 1e-10; % Just to make 1./dx feasible.
Dx = sparse(1:Nx,1:Nx,1./dx); clear dx
Wy = B'*Dx*B;

%%% compute Ncut eigenvectors
% normalized affinity matrix
d = sum(Wy,2);
D = sparse(1:Ny,1:Ny,1./sqrt(d)); clear d
nWy = D*Wy*D; clear Wy
nWy = (nWy+nWy')/2;

% computer eigenvectors
[evec,eval] = eig(full(nWy)); clear nWy   
[~,idx] = sort(diag(eval),'descend');
Ncut_evec = D*evec(:,idx(1:Nseg)); clear D

%%% compute the Ncut eigenvectors on the entire bipartite graph (transfer!)
evec = Dx * B * Ncut_evec; clear B Dx Ncut_evec

% normalize each row to unit norm
evec = bsxfun( @rdivide, evec, sqrt(sum(evec.*evec,2)) + 1e-10 );

% k-means
labels = kmeans(evec,Nseg,'MaxIter',maxKmIters,'Replicates',cntReps);

