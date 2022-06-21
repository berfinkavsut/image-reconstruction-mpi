function [history,result] = admm_tv_nonnegative(A,b,rho,mu,lambda,step_length,MAX_ITER,varargin)
% performs ADMM to solve TV/L2 problem, e: time spent, result: resulted x.
% Original problem:
% minimize (mu/2)*||Ax-b||^2 + lambda* TV(x) such that x_i>0 where x_i^'s
% are i^th element of x vector
%
% With ADMM:
% minimize sum_over_i(||z_i||_2) + (mu/2)*||Ax-b||^2 + i_c(w)
%
% subject to x = w and D_i * x = z_i where i=1,...,num_of_pos
% i_c is the indicator function of whether its argument is in set C or not.
%
% rho: penalty parameter in augmented Lagrangian.
% step_length:step size for dual update,
% reference paper: "ALTERNATING DIRECTION ALGORITHMS FOR TOTAL VARIATION
% DECONVOLUTION IN IMAGE RECONSTRUCTION" by Min TAO and Junfeng YANG

ref_flag = false; % reference image does not exist in default.
if nargin>7 %checks whether reference image is given as input.
 ref_flag = true;

 ref_im = varargin{1};
end

%Global constants and defaults
ABSTOL = 1e-4;
RELTOL = 1e-4;
t = cputime;
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
z = zeros(2*num_pos,1); % First dimension is 2*num_pos since horizontal and

% vertical differences are stacked in a fashion that the first half contains
% the horizontal difference info and the second half contains the vertical
% difference info.
w = x;
y1 = zeros(2*num_pos,1); %dual variable for TV term.
y2 = zeros(num_pos,1); %dual variable for nonnegativity constraint.
  
%% iteration until relative change in x is small enough.
left_multip = inv(rho*(D'*D) + (mu)* (A'*A)+ rho*eye(num_pos));%calculate once, use in every iter

for i = 1:MAX_ITER
 %% x-minimization step.
 right_multip = D'*(rho*z-y1) + mu * A'*b- y2 + rho * w;
34
 x = left_multip * right_multip;
 
 %% z-minimization step.
 z_old = z;
 finitediff_reordered = reshape(D*x,num_pos,2); %each row represents z_i
 y_reordered = reshape(y1,num_pos,2);
 temp = soft_threshold_2D(finitediff_reordered+(1/rho)*y_reordered,lambda/rho);
 z = reshape(temp,2*num_pos,1);
 
 %% dual variable update.
 y1 = y1 - step_length*rho*(z-D*x);
 
 %% w-minimization step.
 w_old = w;
 w = x + (1/rho)*y2;
 w(w<0) = 0;
 
 %% dual variable update.
 y2 = y2 + step_length*rho*(x-w);

 %% history
 history.obj(i)= objective(A,b,x,mu,lambda,D);%calculates the objective
 % function value for current iteration.
 
 if ref_flag
 %calculates nrmse if reference exists
 history.nrmse(i) = sqrt(immse(x,ref_im))/(max(x)-min(x));
 end
 
 %% stopping criteria check.
 residual_primal1 = norm(D*x - z);
 residual_primal2 = norm(x - w);
 residual_dual1 = norm(-rho*D'*(z - z_old));
 residual_dual2 = norm(-rho*(w - w_old));
 
 tolerance_primal1 = sqrt(num_pos)*ABSTOL + RELTOL*max(norm(D*x), norm(z));
 tolerance_primal2 = sqrt(num_pos)*ABSTOL + RELTOL*max(norm(x), norm(w));
 tolerance_dual1 = sqrt(num_pos)*ABSTOL + RELTOL*norm(y1);
 tolerance_dual2 = sqrt(num_pos)*ABSTOL + RELTOL*norm(y2);
 
 if ((residual_primal1 < tolerance_primal1) && (residual_dual1 < tolerance_dual1) && (residual_primal2 < 
tolerance_primal2) && (residual_dual2 < tolerance_dual2))
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
