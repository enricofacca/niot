%% Script for running inpainting tools from 
% https://doi.org/10.5281/zenodo.4315173
% https://github.com/simoneparisotto/MATLAB-Python-inpainting-codes
%
% Authors:
% Enrico Facca              (email:enrico.facca dot uib dot no)
%      
% Date:
% April, 2024
%
% Licence: BSD-3-Clause (https://opensource.org/licenses/BSD-3-Clause)
%
clear
close all
clc

addpath ./MATLAB-Python-inpainting-codes/matlab/lib
%addpath ./dataset


directory = './data/y_net_nref2/';


network_name = 'network_artifacts';
mask_name = 'mask_medium';
corrupted_name = strcat(mask_name,'_',network_name);

example_dir = strcat('results/y_net_nref2/');
out_dir = strcat('results/y_net_nref2/',mask_name,'/');

if ~exist(out_dir, 'dir')
    mkdir(out_dir)
end
if ~exist(example_dir, 'dir')
    mkdir(example_dir)
end


clean_filename = strcat(directory,network_name,'.png');
mask_filename  = strcat(directory,mask_name,'.png');

% function defined at the end of file
[corrupted,mask] = apply_mask(clean_filename,mask_filename);
corrupted_filename = strcat(out_dir,corrupted_name,'.png');
imwrite(corrupted,corrupted_filename)

approaches = ["cahn_hilliard","transport","harmonic","mumford_shah"];

for i = 1 : length(approaches)
    approach = approaches(i);
    rec = reconstruct(corrupted, mask, approach);
    reconstruction_filename = strcat(out_dir,corrupted_name,'_',approach','.png');
    imwrite(rec, reconstruction_filename)
end


function [reconstructed] = reconstruct(corrupted, mask, approaches)
    %% MATLAB Codes for the Image Inpainting Problem
    % This function is adapted from the original code in 
    % https://doi.org/10.5281/zenodo.4315173
    % https://github.com/simoneparisotto/MATLAB-Python-inpainting-codes
    %
    % Authors:
    % Enrico Facca              (email:enrico.facca dot uib dot no)
    %      
    %  
    % Date:
    % April, 2024
    %
    % Licence: BSD-3-Clause (https://opensource.org/licenses/BSD-3-Clause)
    %
    
    
    
    if contains(approaches,'amle')
	    %% AMLE (Absolute Minimizing Lipschitz Extension) Inpainting
    
	    % parameters
	    lambda        = 10^2; 
	    tol           = 1e-8;
	    maxiter       = 40000;
	    dt            = 0.01;
    
	    % inpainting
	    tic
	    reconstructed=inpainting_amle(corrupted,mask,lambda,tol,maxiter,dt);
	    toc
    end
    
    if any(contains(approaches,'harmonic'))
	    %% Harmonic Inpainting
	    
	    % parameters
	    lambda        = 10;
	    tol           = 1e-5;
	    maxiter       = 500;
	    dt            = 0.1;
    
	    % inpainting
	    tic
	    reconstructed = inpainting_harmonic(corrupted,mask,lambda,tol,maxiter,dt);
	    toc
	    %fout=strcat(outdir,test,level,bw,'_harmonic.png')
	    %imwrite(u,fout)
    end
    
    
    if contains(approaches,'mumford_shah')
	    %% Mumford-Shah Inpainting
	    
	    % parameters
	    maxiter       = 20; 
	    tol           = 1e-14;
	    param.lambda  = 10^9;   % weight on data fidelity (should usually be large).
	    param.alpha   = 1;      % regularisation parameters \alpha.
	    param.gamma   = 0.5;    % regularisation parameters \gamma.
	    param.epsilon = 0.05;   % accuracy of Ambrosio-Tortorelli approximation of the edge set.
    
	    % inpainting
	    %tic
	    reconstructed=inpainting_mumford_shah(corrupted,mask,maxiter,tol,param);
	    toc
    end
    
    
    if contains(approaches,'cahn_hilliard')
	    %% Cahn-Hilliard Inpainting
	    
	    % parameters
	    maxiter       = 4000; 
	    param.epsilon = [100 1];
	    param.lambda  = 10;
	    param.dt      = 1;
    
	    % inpainting
	    tic
	    reconstructed = inpainting_cahn_hilliard(corrupted,mask,maxiter,param);
	    toc
    end
    
    if contains(approaches,'transport')
    
	    %% Transport Inpainting
	    
	    % parameters
	    tol           = 1e-5;
	    maxiter       = 50;
	    dt            = 0.1;
	    param.M       = 40; % number of steps of the inpainting procedure;
	    param.N       = 2;  % number of steps of the anisotropic diffusion;
	    param.eps     = 1e-10;
    
	    % inpainting
	    tic
	    reconstructed = inpainting_transport(corrupted,mask,maxiter,tol,dt,param);
	    toc
    end

end


function [u,mask,input] = apply_mask(imagefilename,maskfilename)

    % import a clean input to be corrupted with the mask 
    input = im2double(imread(imagefilename));

    % import the mask of the inpainting domain
    % mask = 1 missing domain
    % mask = 0 
    mask  = double( mat2gray( im2double(imread(maskfilename)) ) == 1 );
    if size(mask,3)==1 && size(input,3)>1
        mask = repmat(mask,[1,1,size(input,3)]);
    end

    % create the image with the missin domain:
    u     = (mask).*input+(1-mask);% + (1-mask).*noise;
end