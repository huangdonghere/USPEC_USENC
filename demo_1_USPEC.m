%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% This is a demo for the U-SPEC algorithm, which is proposed in the %
% following paper:                                                  %
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

function demo_1_USPEC()
%% Run the USPEC algorithm multiple times and show its average performance.

clear all;
close all;

%% Load the data.
% Please uncomment the dataset that you want to use and comment the other ones.
dataName = 'PenDigits';     % Real datasets
% dataName = 'USPS';
% dataName = 'Letters';
% dataName = 'MNIST';
% dataName = 'Covertype';
% dataName = 'TB1M';        % Synthetic datasets
% dataName = 'SF2M';
% dataName = 'CC5M';
% dataName = 'CG10M';
% dataName = 'Flower20M';

% Load the dataset.
dataNameFull = ['data_',dataName,'.mat'];
if ~exist(dataNameFull)
    if strcmp(dataName,'TB1M') || strcmp(dataName,'SF2M') || strcmp(dataName,'CC5M') || strcmp(dataName,'CG10M') || strcmp(dataName,'Flower20M')
        synthesizeLargescaleDatasets(dataName);
        pause(0.01);
    else
        disp('The dataset doesn''t exist!');
        return;
    end
end
gt = [];
fea = [];
load(['data_',dataName,'.mat'],'fea','gt'); 

[N, d] = size(fea);

%% Set up
k = numel(unique(gt)); % The number of clusters
cntTimes = 20; % The number of times that the USPEC algorithm will be run.


%% Run USPEC
nmiScores = zeros(cntTimes,1);
disp('.');
disp(['N = ',num2str(N)]);
disp('.');
for runIdx = 1:cntTimes
    disp('**************************************************************');
    disp(['Run ', num2str(runIdx),':']);
    disp('**************************************************************');
    
    disp('.');
    disp('Performing U-SPEC ...');
    disp('.');
    
    % You can use the default parameters (p=1000, KNN=5)
%     tic;
%     Label = USPEC(fea, k);
%     toc;
    
    % Or you can set up parameters by yourself.
    tic;
    p = 1000; % Number of representatives
    KNN = 5; % Number of nearest neighbors
    Label = USPEC(fea, k, p, KNN);
    toc;
    
    disp('.');
    
    disp('--------------------------------------------------------------');
    nmiScores(runIdx) = computeNMI(Label,gt);
    disp(['The NMI score at Run ',num2str(runIdx), ': ',num2str(nmiScores(runIdx))]);   
    disp('--------------------------------------------------------------');
end

disp('**************************************************************');
disp(['  ** Average Performance over ',num2str(cntTimes),' runs on the ',dataName,' dataset **']);
disp(['Sample size: N = ', num2str(N)]);
disp(['Dimension:   d = ', num2str(d)]);
disp('--------------------------------------------------------------');
disp(['Average NMI score: ',num2str(mean(nmiScores))]);
disp('--------------------------------------------------------------');
disp('**************************************************************');
