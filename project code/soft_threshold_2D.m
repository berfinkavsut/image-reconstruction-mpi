% Used inside admm functions.

function res = soft_threshold_2D(arg,threshold)
%
% 2D shrinkage operator.
%
res = max(0,vecnorm(arg,2,2)-threshold) .* arg./(vecnorm(arg,2,2));
res(isnan(res))=0;
end
