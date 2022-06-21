% only shown for dataset of concentration phantom
% this code is used for choosing optimal lambda values for regularized weighted kaczmarz solution 
clear all;
close all; 

%load system matrix and measurement vector 
load('SM_website.mat','S');
load('meas_website.mat','u');

%SVD of system matrix  
[U,Sigma,V] = svd(S,'econ');

%condition number of Sigma matrix 
condition_no = cond(Sigma); 

%row norm thresholding
[size_original,~] = size(S);
threshold = 50;
[S,u] = row_norm_treshold(threshold,S,u);

%SVD of new matrix 
[size_threshold,~] = size(S);
[U,Sigma,V] = svd(S,'econ');
condition_no_threshold= cond(Sigma); 

residual_norm = zeros(length(lambda),1);
solution_norm = zeros(length(lambda),1);

opt = 'other';
lambda= 1e3:1000:1e6;

%find residual norms and solution norms 
for k = 1:length(lambda)
 [c,history] = filtered_svd(S,u,lambda(k),opt);
 residual_norm(k)= norm(S*c-u,2);
 solution_norm(k) = norm(c,2);
end 

%draw L-curve 
figure;set(gcf, 'WindowState', 'maximized');
plot(residual_norm,solution_norm);
xlabel('||Sc-u||');ylabel('||c||');
title('L-curve');
saveas(gcf,'Lcurve_website.png');

%solution_norm = 27.21 and residual_norm = 3330 for lambda_optimal
[~,index] = min(abs(solution_norm-27.21));
website.lambda_optimal = lambda(index);
%lambda_optimal = 11000;
