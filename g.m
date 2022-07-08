function [fun,grad] = g(x,ind,L,gamma,tau)
% Auxiliary function constituting octopus
% Input: ( x, index i, L, gamma, tau ) 
% x is a vector
% Output: function value of g, and gradient vector whose only nonzero
% components are ind, ind+1

dim = length(x);
xi = x(ind);
xj = x(ind+1);

para = L+gamma;

g1 = -gamma*xi^2 + (-14*L+10*gamma)/(3*tau)*(xi-tau)^3 +...
    (5*L-3*gamma)/(2*tau^2) *(xi-tau)^4;
g2 = -gamma - 10*para/tau^3 * (xi-2*tau)^3 -...
    15*para/tau^4 *(xi-2*tau)^4 - 6*para/tau^5 *(xi-2*tau)^5;
fun = g1 + g2 * xj^2;

grad = zeros(dim,1);
grad(ind) = (-2*gamma*xi + (-14*L+10*gamma)/tau*(xi-tau)^2 + ...
    (10*L-6*gamma)/tau^2 * (xi-tau)^3)+...
    ( -30*para/tau^3*(xi-2*tau)^2 - 60*para/tau^4 *(xi-2*tau)^3 -...
    30*para/tau^5*(xi-2*tau)^4 )*xj^2;
grad(ind+1) = g2*2*xj;

end