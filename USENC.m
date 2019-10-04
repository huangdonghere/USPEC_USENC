%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% This is the code for the U-SPEC algorithm, which is proposed in   %
% the following paper:                                              %
%                                                                   %
% D. Huang, C.-D. Wang, J.-S. Wu, J.-H. Lai, and C.-K. Kwoh.        %
% "Ultra-Scalable Spectral Clustering and Ensemble Clustering."     %
% To appear in IEEE TKDE, 2019.                                     %
% DOI: https://doi.org/10.1109/TKDE.2019.2903410                    %
%                                                                   %
% The code has been tested in Matlab R2016a and Matlab R2016b.      %
% Website: https://www.researchgate.net/publication/330760669       %
% Written by Huang Dong. (huangdonghere@gmail.com)                  %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function L = USENC(fea, k, M, p, Knn, bcsLowK, bcsUpK)
if nargin < 7
    bcsUpK = 60;
end
if nargin < 6
    bcsLowK = 20;
end
if nargin < 5
    Knn = 5;
end
if nargin < 4
    p = 1000;
end
if nargin < 3
    M = 20;
end
if bcsUpK < bcsLowK
    bcsUpK = bcsLowK;
end

disp('.');
disp(['Generating an ensemble of ',num2str(M),' base clusterings...']);
disp('.');
baseCls = USENC_EnsembleGeneration(fea, M, p, Knn, bcsLowK, bcsUpK);
clear fea

disp('.');
disp('Performing the consensus function...');
disp('.');
tic1 = tic;
L = USENC_ConsensusFunction(baseCls,k);
toc(tic1);
disp('.');

function members = USENC_EnsembleGeneration(fea, M, p, Knn, lowK, upK)
% Huang Dong. Mar. 20, 2019.
% Generate M base cluserings.
% The number of clusters in each base clustering is randomly selected in
% the range of [lowK, upK].
if nargin < 6
    upK = 60;
end
if nargin < 5
    lowK = 20;
end
if nargin < 4
    Knn = 5;
end
if nargin < 3
    p = 1000;
end
if nargin < 2
    M = 20;
end

N = size(fea,1);
if p>N
    p = N;
end
members = zeros(N,M);

rand('state',sum(100*clock)*rand(1)); % Reset the clock before generating random numbers
Ks = randsample(upK-lowK+1,M,true)+lowK-1;

warning('off');
% In ensemble generation, the iteration number in the kmeans discretization 
% of each base cluserer can be set to small values, so as to improve
% diversity of base clusterings and reduce the iteration time costs.
tcutKmIters = 5;
tcutKmRps = 1;
for i = 1:M
    % Generating the i-th base clustering by U-SPEC.
    tic1 = tic;
    members(:,i) = USPEC(fea, Ks(i), p, Knn, tcutKmIters, tcutKmRps);
    toc(tic1);
end


function labels = USENC_ConsensusFunction(baseCls,k,maxTcutKmIters,cntTcutKmReps)
% Huang Dong. Mar. 20, 2019.
% Combine the M base clusterings in baseCls to obtain the final clustering
% result (with k clusters).

if nargin < 4
    cntTcutKmReps = 3; 
end
if nargin < 3
    maxTcutKmIters = 100; % maxTcutKmIters and cntTcutKmReps are used to limit the iterations of the k-means discretization in Tcut.
end

[N,M] = size(baseCls);

maxCls = max(baseCls);
for i = 1:numel(maxCls)-1
    maxCls(i+1) = maxCls(i+1)+maxCls(i);
end

cntCls = maxCls(end);
baseCls(:,2:end) = baseCls(:,2:end) + repmat(maxCls(1:end-1),N,1); clear maxCls

% Build the bipartite graph.
B=sparse(repmat([1:N]',1,M),baseCls(:),1,N,cntCls); clear baseCls
colB = sum(B);
B(:,colB==0) = [];

% Cut the bipartite graph.
labels = Tcut_for_bipartite_graph(B,k,maxTcutKmIters,cntTcutKmReps);



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
% labels = kmeans(evec,Nseg);
labels = kmeans(evec,Nseg,'MaxIter',maxKmIters,'Replicates',cntReps);
