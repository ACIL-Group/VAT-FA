% This is an example of usage of the VAT + Fuzzy ART Framework.
%
% PROGRAM DESCRIPTION
% This program exemplifies the usage of the VAT + Fuzzy ART code provided.
%
% REFERENCES
% [1] L. E. Brito da Silva and Donald C. Wunsch II, A study on exploiting 
% VAT to mitigate ordering effects in Fuzzy ART, in IEEE 2018 International 
% Joint Conference on Neural Networks (IJCNN 2018).
% [2] G. Carpenter, S. Grossberg, and D. Rosen, "Fuzzy ART: Fast 
% stable learning and categorization of analog patterns by an adaptive 
% resonance system," Neural networks, vol. 4, no. 6, pp. 759–771, 1991.
% [3] J. C. Bezdek and R. J. Hathaway, “VAT: a tool for visual assessment
% of (cluster) tendency,” in The 2002 International Joint Conference on
% Neural Networks (IJCNN), vol. 3, May 2002, pp. 2225–2230.
%
% Code written by Leonardo Enzo Brito da Silva
% Under the supervision of Dr. Donald C. Wunsch II
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%% Clean Run
clear variables; close all; echo off; clc;

%% Add path
addpath('classes', 'functions');

%% Load data
load('clusterdemo.dat')  % Data set available from MATLAB's Fuzzy Logic Toolbox
data = clusterdemo;
[nSamples, dim] = size(data);

% Linear Normalization
data = mapminmax(data', 0, 1);
data = data';

% Randomize
seed = 12345; % for reproducibility
generator = 'twister';        
rng(seed, generator); 
Prng = randperm(nSamples);
data_rng = data(Prng, :);

%% ART Parameter settings
settings = struct();
settings.rho = 0.41;
settings.alpha = 1e-3;
settings.beta = 1;
nEpochs = 2;

%% Fuzzy ART
% Train
FA_rng = FuzzyART(settings);
FA_rng = FA_rng.train(data_rng, nEpochs); 

%% VAT + Fuzzy ART                         
M = pdist2(data_rng, data_rng);          
[D, Pvat] = VAT(M);
data_vat = data_rng(Pvat, :);

% Train
FA_vat = FuzzyART(settings);
FA_vat = FA_vat.train(data_vat, nEpochs); 

%% ART properties
clc
FA_rng
FA_vat
%% Scatterplot of Clusters
% RNG
clrs_rng = rand(FA_rng.nCategories, 3);
C_rng = clrs_rng(FA_rng.labels, :);
% VAT
clrs_vat = rand(FA_vat.nCategories, 3);
C_vat = clrs_vat(FA_vat.labels, :);

% Figures
figure
imagesc(D)
title('VAT')
colorbar
axis square

figure
S = 30;
subplot(1,2,1)
scatter3(data_rng(:, 1), data_rng(:, 2), data_rng(:, 3), S, C_rng, 'filled') 
title('Fuzzy ART')
view([145 26]);
grid on
axis square
subplot(1,2,2)
scatter3(data_vat(:, 1), data_vat(:, 2), data_vat(:, 3), S, C_vat, 'filled') 
title('VAT + Fuzzy ART')
view([145 26]);
grid on
axis square