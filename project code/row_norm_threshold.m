function [S,u] = row_norm_treshold(treshold,S,u) 
[row_no,~] = size(S); 
norm_S = sqrt(sum(S.^2,2)); 
index = find(norm_S<treshold);
S(index,:) = [];
u(index,:) = [];
end
