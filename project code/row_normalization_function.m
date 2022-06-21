function [S,u] = row_normalization(S,u)
 %row normalization
 row_energy = sum(S.^2,2);
 w = 1./row_energy;
 W = diag(w);
 S = sqrt(W)*S;
 u = sqrt(W)*u;
end
