function [history, res] = admm_lasso(A,b,rho,mu,tau,step_length,MAX_ITER,varargin)
% performs ADMM to solve LASSO problem, returns history struct and resulted
% vector x.
% Original problem:
% minimize (mu/2)*||Ax-b||^2 + tau * ||x||_1
%
% With ADMM :
% minimize tau * ||z||_1 + (mu/2)*||Ax-b||^2
%
% subject to x = z;
%
% rho : penalty parameter, step_length: step size for dual update,
% variable length depends on the whether reference image is given or not.
%
% reference paper: "Distributed Optimization and Statistical Learning via
% the Alternating Direction Method of Multipliers" by Stephen Boyd, Neal
% Parikh, Eric Chu, Borja Peleato and Jonathan Eckstein.
  
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
z = x; %splitted variable for l1 norm constraint.
y = zeros(num_pos,1); %dual variable of z.
t = cputime;
left_multip = inv(mu*(A'*A) + rho* eye(num_pos)); % compute only once.
                      
for i= 1:MAX_ITER

 %% x-update.
 right_multip = mu * A'*b + rho*z-y;
 x = left_multip*right_multip;
 
 %% z-update.
 z_old = z;
 z = soft_threshold(x+(1/rho)*y,tau/rho);
 
 %% dual update.
 y = y + step_length * rho*(x-z);
 
 %% history
 history.obj(i)= objective(A,b,x,mu,tau); %calculates the objective
 % function value in each iteration
 
 if ref_flag
 %calculates nrmse if reference image exists.
 history.nrmse(i) = sqrt(immse(x,ref_im))/(max(x)-min(x));
 end
 
 %% stopping criteria check.
 residual_primal = norm(x - z);
 residual_dual = norm(-rho*(z - z_old));
 tolerance_primal = sqrt(num_pos)*ABSTOL + RELTOL*max(norm(x), norm(z));
 tolerance_dual = sqrt(num_pos)*ABSTOL + RELTOL*norm(y);
 
 if ((residual_primal < tolerance_primal) && (residual_dual < tolerance_dual))
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
