function val = g1(x,L,gamma,tau,flag)
% Auxiliary function to compute octopus
% Input: g1( x, L, gamma, tau)
% x is a scalar corresponding to x_i in the paper
% flag = 1 output g1 function value; flag = 0 output g1 derivative value
if flag == 1 % output function vallue
    val = -gamma*x^2 + (-14*L+10*gamma)/(3*tau)*(x-tau)^3 + (5*L-3*gamma)/(2*tau^2)*(x-tau)^4;
end

if flag == 2 % output gradient
    val = -2*gamma*x + (-14*L+10*gamma)/tau*(x-tau)^2 + (10*L-6*gamma)/tau^2 *(x-tau)^3;
end

end