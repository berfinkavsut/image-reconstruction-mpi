function [c,history]= regularized_kaczmarz(S,u,lambda,iter,ref,opt)

t = cputime;
[N,M] = size(S);
row_energy = sum(S.^2,2);

switch opt 
 case 'uniform'
 w = ones(N,1);
 W = eye(N);
 case 'normalized'
 %weighting function 
 w = 1./row_energy;
 W = diag(w);
end

identity = eye(N);
c = zeros(M,1);
v = zeros(N,1);
alpha = 0;

for k = 1:iter 
 
 %randomized sub-iterations 
 iter_order = randperm(N);

 for i = 1:N
 ind = iter_order(i);
 alpha = ( u(ind) - dot(S(ind,:)',c) - sqrt(lambda/w(ind) ) * v(ind) ) /...
 ( row_energy(ind) + lambda/w(ind));
 c = c + alpha*(S(ind,:)');
 ei = identity(:,ind);
 v = v + alpha * sqrt(lambda/w(ind)) * ei ;
 end
 
 %enforce positive values on image
 %c(c<0) = 0;
 
 history.ima(:,k) = c; 
 history.nrmse(k) = sqrt(immse(c,ref))/(max(c)-min(c));
 history.weighting_option= opt;
 history.regularization_parameter = lambda; 
 history.obj(k) = objective(S,u,c,W,lambda);
end

history.time = cputime-t;
                
end

function obj = objective(S,u,c,W,lambda)
 obj = norm(sqrt(W)*(S*c-u),2)^2 + lambda*(norm(c,2))^2;
end
