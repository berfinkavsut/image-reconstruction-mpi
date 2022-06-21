%Used inside admm functions.

function [res] = soft_threshold(arg,threshold)
%
% 1D shrinkage operator.
%
res = max(0,abs(arg)-threshold) .* arg./(abs(arg));
res(isnan(res))=0;
end
