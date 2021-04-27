function yval = huber(x,omega)
% Huber function
% input: x: vector, omega: huber parameter, the larger it is, the closer it
% is to l1-penalty
% output: y
    ind_set = abs(x)<=1/omega;
    ind_set_c = setdiff(1:1:length(x),ind_set);
    y(ind_set) = omega * x(ind_set).^2;
    y(ind_set_c) = abs(x(ind_set_c)) - 1/(2*omega);
    yval = sum(y);
end

