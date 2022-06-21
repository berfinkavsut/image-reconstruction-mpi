function [history, res] = admm_lasso_nonnegative(A,b,rho,mu,tau,step_length,MAX_ITER,varargin)
% performs ADMM to solve nonnegative LASSO problem, returns history struct 
% and resulted vector x.
% Original problem:
% minimize (mu/2)*||Ax-b||^2 + tau * ||x||_1 such that x_i>0 where x_i^'s
% are i^th element of x vector
%
% With ADMM :
% minimize tau * ||z||_1 + (mu/2)*||Ax-b||^2 + i_c(w)
% subject to x = z and x = w;
% i_c is the indicator function of whether its argument is in set C or not.
%
% rho : penalty parameter, step_length: step size for dual update.
% variable length depends on the whether reference image is given or not.

  ref_flag = false; % reference image does not exist in default.
if nargin>7 %checks whether reference image is given as input.
 ref_flag = true;
 ref_im = varargin{1};
end

%Global constants and defaults
ABSTOL = 1e-4;
RELTOL = 1e-4;
[~, num_pos] = size(A);
x = zeros(num_pos,1); %primal variable.
z = x; %splitted variable associated with l1 norm
w = x; %splitted variable associated with nonnegativity constraint.
y1 = zeros(num_pos,1); %dual variable of z.
y2 = zeros(num_pos,1); %dual variable of w.
t = cputime;
left_multip = inv(mu*(A'*A) + 2*rho* eye(num_pos)); % compute only once.
                      
for i = 1:MAX_ITER

 %% x-update.
 right_multip = mu * A'*b + rho*(z+w)-y1-y2;
 temp = x;
 x = left_multip*right_multip;
 
 %% z-update. (term associated with l1 norm)
 z_old = z;
 z = soft_threshold(x+(1/rho)*y1,tau/rho);
 
 %% dual update.
 y1 = y1+ step_length * rho*(x-z);
 
 %% w-update. (term associated with nonnegative constraint)
 w_old = w;
 w = x+(1/rho)*y2;
 w(w<0) = 0;
 
 %% dual update.
 y2 = y2+step_length*rho*(x-w);
 
 %% history
 history.obj(i)= objective(A,b,x,mu,tau); %calculates the objective
 % function value in each iteration.
 
 if ref_flag
 %calculates nrmse if reference image exists.
 history.nrmse(i) = sqrt(immse(x,ref_im))/(max(x)-min(x));
 end
 
 %% stopping criteria check.
 residual_primal1 = norm(x - z);
 residual_primal2 = norm(x - w);
 residual_dual1 = norm(-rho*(z - z_old));
 residual_dual2 = norm(-rho*(w - w_old));
 
 tolerance_primal1 = sqrt(num_pos)*ABSTOL + RELTOL*max(norm(x), norm(z));
 tolerance_primal2 = sqrt(num_pos)*ABSTOL + RELTOL*max(norm(x), norm(w));
 tolerance_dual1 = sqrt(num_pos)*ABSTOL + RELTOL*norm(y1);
 tolerance_dual2 = sqrt(num_pos)*ABSTOL + RELTOL*norm(y2);
 
 if ((residual_primal1 < tolerance_primal1) && (residual_dual1 < tolerance_dual1) && ...
 (residual_primal2 < tolerance_primal2) && (residual_dual2 < tolerance_dual2))
 break;
 end
end

res = x;
history.cpu_time = cputime-t;
                      
end

function res = objective(A,b,x,mu,tau)
%calculates LASSO objective
res = (mu/2)*norm(A*x-b)^2 + tau * norm(x,1);
end
