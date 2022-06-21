function [c,history] = filtered_svd (S,u,lambda,opt)
  
t = cputime;

[U,Sigma,V] = svd(S,'econ');

switch opt
 case 'singular'
 singular_values = diag(Sigma);
 lambda = singular_values(1)* singular_values(end);
 case 'other'
 %do nothing
end

%solution
new_Sigma = Sigma + (lambda./Sigma);
c = V*((U'*u)./diag(new_Sigma));
        
history.time = cputime-t; 
history.regularization_parameter = lambda; 
        
end
