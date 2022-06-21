function [history,res] = admm_tv_sparse_wo_inverse(A,b,rho,mu,lambda,tau,step_length,MAX_ITER,varargin)
% admm_tv_sparse(A,b,rho,mu,lambda,step_length,tolerance)
% ADMM Function with Total Variation and L1 norm regularization
% performs ADMM to find a sparse solution to the TV/L2 problem. e:time
% spent, res: resulted x vector.
%
% Original problem:
% minimize (mu/2)*||y - Mx||^2 + tau*||x||_1 + lambda*||x||_TV
%
% With ADMM:
% minimize (mu/2)*||y - Mx||^2 + tau*||v||_1 + lambda*||w||_2
% such that v = x, w_i = D_i*x for i = 1...n_y
% D_i*x Rd represents the first-order finite difference of x at i:th component in d different directions.
%
% rho: penalty parameter in augmented Lagrangian.
% step_length:step size for dual update,
% tolerance: value for relative change in x and used as stopping criteria.
% 
% reference paper: "Combined Analysis-L1 and Total Variation ADMM with 
% Applications to MEG Brain Imaging and Signal Reconstruction" by Rui Gao, 
% Filip Tronarp, and Simo Särkkä

ref_flag = false; % reference image does not exist in default.
if nargin>8 %checks whether reference image is given as input.
 ref_flag = true;
 ref_im = varargin{1};
end

%Global constants and defaults
ABSTOL = 1e-4;
RELTOL = 1e-4;
t_start = cputime;
[~, num_pos] = size(A);
grid_size = sqrt(num_pos);

%% compute the difference matrix D_i for each i 
% For the point (i,j), take the difference between (i+1,j) and (i,j+1)
% horizontal difference matrix.
n = num_pos;
e = ones(n,1);
D_horizontal = spdiags([e -e],0:1,n,n);

% horizontal finite difference can not be calculated for the pixels on the
% right edge. So, make the operator zero for that pixels.
%border_ind = grid_size * (1:grid_size);
D_horizontal(1,:) = 0;
D_horizontal(end,:) = 0;

%vertical difference matrix.
e = ones(n,1);
D_vertical = spdiags([-1*e e],[0,grid_size],n,n);

% vertical finite difference can not be calculated for the pixels on the
% bottom edge. So, make the operator zero for that pixels.
%border_ind = grid_size * (grid_size-1);
D_vertical(:,1) = 0;
D_vertical(:,end) = 0;
D = [D_horizontal', D_vertical']';
  
%% ADMM Steps
delta = 1/((rho + rho * normest(D)^2 + mu * norm(A,2)^2));
x = zeros(num_pos,1);
w = zeros(num_pos,1); %sparsity 
z = zeros(2*num_pos,1); %TV
y1 = zeros(num_pos,1);
y2 = zeros(2*num_pos,1); 

for i = 1:MAX_ITER

 %% x-update
 x = x - delta*(y1 + D'*y2)+delta*rho*(x-w)+delta*rho*D'*(D*x-z)+delta*mu*A'*(b-A*x);
                
 %% w-update
 w_old = w;
 e = x + y1/rho;
 w = e./abs(e).*max(abs(e)-tau/rho,0); %Enforce sparsity
 
 %% z-update
 z_old = z;
 t = reshape(D*x,num_pos,2) + reshape(y2,num_pos,2)/rho;
 z = reshape(max(norm(t,2)-lambda/rho,0).*t/norm(t,2),2*num_pos,1); %TV term update
 
 %% dual variable for sparsity constraint update (eta)
 y1 = y1 + step_length*rho*(x - w);
                
 %% dual variable for TV update (ksi)
 y2 = y2 + step_length*rho*(D*x - z); %ksi update
 
 %% history
 history.obj(i)= objective(A,b,x,mu,lambda,tau,D);%calculates the 
 
 % objective function value for current iteration 
 if ref_flag %calculates nrmse if reference exists
 history.nrmse(i) = sqrt(immse(x,ref_im))/(max(x)-min(x));
 end
 
 %% stopping criteria check.
 residual_primal1 = norm(D*x - z);
 residual_primal2 = norm(x - w);
 residual_dual1 = norm(-rho*D'*(z - z_old));
 residual_dual2 = norm(-rho*(w - w_old));
 
 tolerance_primal1 = sqrt(num_pos)*ABSTOL + RELTOL*max(norm(D*x), norm(z));
 tolerance_primal2 = sqrt(num_pos)*ABSTOL + RELTOL*max(norm(x), norm(w));
 tolerance_dual1 = sqrt(num_pos)*ABSTOL + RELTOL*norm(y2);
 tolerance_dual2 = sqrt(num_pos)*ABSTOL + RELTOL*norm(y1);
 
 if ((residual_primal1 < tolerance_primal1) && (residual_dual1 < tolerance_dual1) && (residual_primal2 < 
tolerance_primal2) && (residual_dual2 < tolerance_dual2))
 break;
 end 
 
end

res = x;
history.cpu_time = cputime-t_start;
              
end

function res = objective(A,b,x,mu,lambda,tau,D)
%calculates objective least square function with TV and l1 regularization.
res = (mu/2)*norm(A*x-b)^2 + lambda * sum(vecnorm(D*x,2,2))+ tau*norm(x,1);
end
