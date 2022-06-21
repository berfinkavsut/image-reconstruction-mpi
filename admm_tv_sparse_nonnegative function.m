function [history,res] = admm_tv_sparse_nonnegative(A,b,rho,mu,lambda,tau,step_length,MAX_ITER,varargin)
% performs ADMM to find a sparse solution to the TV/L2 problem. returns 
% history struct and resulted vector x.
%
% Original problem:
% minimize (mu/2)*||Ax-b||^2 + lambda* TV(x) + tau * ||x||_1 such that x_i>0 
% where x_i^'s are i^th element of x vector
%
% With ADMM:
% minimize (mu/2)*||Ax-b||^2 + sum_over_i(||z_i||_2) + tau*||w||_1 + i_c(s)
%
% subject to D_i * x = z_i where i=1,...,num_of_pos and w = x and s = x
% i_c is the indicator function of whether its argument is in set C or not.
%
% rho: penalty parameter in augmented Lagrangian.
% step_length:step size for dual update,
% variable length depends on the whether reference image is given or not.

ref_flag = false; % reference image does not exist in default.
if nargin>8 %checks whether reference image is given as input.
 ref_flag = true;
 ref_im = varargin{1};
end

%Global constants and defaults
ABSTOL = 1e-4;
RELTOL = 1e-3;
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
x = zeros(num_pos,1); %primal variable.
w = x; %auxilary variable for sparsity constraint.
y1 = zeros(num_pos,1); %dual variable of w.
z = zeros(2*num_pos,1); % First dimension is 2*num_pos since horizontal and

% vertical differences are stacked in a fashion that the first half contains
% the horizontal difference info and the second half contains the vertical
% difference info.
y2 = zeros(2*num_pos,1); %dual variable of z.
s = x; %primal term associated with nonnegativity constraint.
y3 = zeros(num_pos,1); %dual variable of s.
t = cputime;
left_multip = inv(rho*(D'*D) + 2*rho*eye(num_pos) + (mu)*(A'*A)); %compute once.

for i = 1:MAX_ITER
 %% x-update.
 right_multip = (mu)* A'*b + D'*(rho*z-y2) - y1 - y3 + rho*(w + s);
 x = left_multip * right_multip;
 
 %% w-update. (enforce sparsity)
 w_old = w;
 w = soft_threshold( x + (1/rho)*y1,tau/rho); %1D shrinkage operator
 
 %% y1 update (dual variable of sparsity constraint i.e w = x)
 y1 = y1 - step_length * rho * (w - x);
 
 %% z-update. (TV term)
 z_old = z;
 finitediff_reordered = reshape(D*x,num_pos,2); %each row represents z_i
 y2_reordered = reshape(y2,num_pos,2);
 z = reshape(soft_threshold_2D(finitediff_reordered+(1/rho)*y2_reordered,lambda/rho),2*num_pos,1);
 
 %% y2 update (dual variable of TV constraint i.e z_i = D_i * x)
 y2 = y2 - step_length * rho * (z - D * x);
 
 %% s-update (nonnegativity constraint).
 s_old = s;
 s = x + (1/rho)*y3;
 s(s<0) = 0;
 
 %% y3 dual variable update.
 y3 = y3 + step_length*rho*(x-w);
 
 %% history
 history.obj(i)= objective(A,b,x,mu,lambda,tau,D);%calculates the 
 % objective function value for current iteration
 
 if ref_flag %calculates nrmse if reference exists
 history.nrmse(i) = sqrt(immse(x,ref_im))/(max(x)-min(x));
 end
 
 %% stopping criteria check.
 residual_primal1 = norm(D*x - z);
 residual_primal2 = norm(x - w);
 residual_primal3 = norm(x - s);
 residual_dual1 = norm(-rho*D'*(z - z_old));
 residual_dual2 = norm(-rho*(w - w_old));
 residual_dual3 = norm(-rho*(s - s_old));
 
 tolerance_primal1 = sqrt(num_pos)*ABSTOL + RELTOL*max(norm(D*x), norm(z));
 tolerance_primal2 = sqrt(num_pos)*ABSTOL + RELTOL*max(norm(x), norm(w));
 tolerance_primal3 = sqrt(num_pos)*ABSTOL + RELTOL*max(norm(x), norm(s));
 tolerance_dual1 = sqrt(num_pos)*ABSTOL + RELTOL*norm(y2);
 tolerance_dual2 = sqrt(num_pos)*ABSTOL + RELTOL*norm(y1);
 tolerance_dual3 = sqrt(num_pos)*ABSTOL + RELTOL*norm(y3);
 
 if ((residual_primal1 < tolerance_primal1) && (residual_dual1 < tolerance_dual1) && (residual_primal2 < 
tolerance_primal2) && (residual_dual2 < tolerance_dual2) && (residual_primal3 < tolerance_primal3) && 
(residual_dual3 < tolerance_dual3))
 break;
 end
 
end

res = x;
history.cpu_time = cputime-t;
                       
end

function res = objective(A,b,x,mu,lambda,tau,D)
%calculates objective least square function with TV and l1 regularization.
res = (mu/2)*norm(A*x-b)^2 + lambda * sum(vecnorm(D*x,2,2))+ tau*norm(x,1);
end
