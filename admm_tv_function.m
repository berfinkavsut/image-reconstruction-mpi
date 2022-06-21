function [history,result] = admm_tv(A,b,rho,mu,lambda,step_length,MAX_ITER,varargin)
% performs ADMM to solve TV/L2 problem, returns history struct and resulted
% vector x.
% Original problem:
% minimize (mu/2)*||Ax-b||^2 + lambda* TV(x)
%
% With ADMM:
% minimize sum_over_i(||z_i||_2) + (mu/2)*||Ax-b||^2
%
% subject to D_i * x = z_i where i=1,...,num_of_pos
%
% rho: penalty parameter in augmented Lagrangian.
% step_length:step size for dual update,
% variable length depends on the whether reference image is given or not.
%
% reference paper: "ALTERNATING DIRECTION ALGORITHMS FOR TOTAL VARIATION
% DECONVOLUTION IN IMAGE RECONSTRUCTION" by Min TAO and Junfeng YANG
  
ref_flag = false; % reference image does not exist in default.
if nargin>7 %checks whether reference image is given as input.
 ref_flag = true;
 ref_im = varargin{1};
end

%Global constants and defaults
ABSTOL = 1e-5;
RELTOL = 1e-4; 
[~, num_pos] = size(A);
size_1D = sqrt(num_pos);

%% construct difference matrix both in horizontal and vertical direction.
%horizontal difference matrix.
n = num_pos;
e = ones(n,1);
D1 = spdiags([-1*e e],0:1,n,n);

% horizontal finite difference can not be calculated for the pixels on the
% right edge. So, make the operator zero for that pixels.
border_ind = size_1D * (1:size_1D);
D1(border_ind,:) = 0;

%vertical difference matrix.
e = ones(n,1);
D2 = spdiags([-1*e e],[0,size_1D],n,n);

% vertical finite difference can not be calculated for the pixels on the
% bottom edge. So, make the operator zero for that pixels.
border_ind = size_1D * (size_1D-1);
D2(border_ind+1:end,:) = 0;
D = [D1', D2']';
  
%% initialization of x,z,y
x = zeros(num_pos,1); %primal variable.
z = zeros(2*num_pos,1); % spliited variable associated with TV term. 
  
% First dimension is 2*num_pos since horizontal and vertical differences are
% stacked in a fashion that the first half contains the horizontal difference
% info and the second half contains the vertical
% difference info.
y = zeros(2*num_pos,1); %dual variable of z.
t = cputime;
left_multip = inv(rho*(D'*D) + (mu)* (A'*A));%calculate once, use in every iter

for i = 1:MAX_ITER

 %% x-minimization step.
 right_multip = D'*(rho*z-y) + mu * A'*b;
 x = left_multip * right_multip;
 
 %% z-minimization step.
 z_old = z;
 finitediff_reordered = reshape(D*x,num_pos,2); %each row represents z_i
 y_reordered = reshape(y,num_pos,2);
 temp = soft_threshold_2D(finitediff_reordered+(1/rho)*y_reordered,lambda/rho);
 z = reshape(temp,2*num_pos,1);
 
 %% dual variable update.
 y = y - step_length*rho*(z-D*x);
 
 %% history
 history.obj(i)= objective(A,b,x,mu,lambda,D);%calculates the objective
 % function value for current iteration.
 
 if ref_flag %calculates nrmse if reference exists
 history.nrmse(i) = sqrt(immse(x,ref_im))/(max(x)-min(x));
 end
 
 %% stopping criteria check.
 residual_primal = norm(D*x - z);
 residual_dual = norm(-rho*D'*(z - z_old));
 
 tolerance_primal = sqrt(num_pos)*ABSTOL + RELTOL*max(norm(D*x), norm(z));
 tolerance_dual = sqrt(num_pos)*ABSTOL + RELTOL*norm(y);
 
 if ((residual_primal < tolerance_primal) && (residual_dual < tolerance_dual))
 break;
 end
 
end

result = x;
history.cpu_time = cputime-t;
                      
end

function res = objective (A,b,x,mu,lambda,D)
% calculates objective function for TV regularized least square.
res = (mu/2)*norm(A*x-b)^2 + lambda * sum(vecnorm(D*x,2,2));
end
