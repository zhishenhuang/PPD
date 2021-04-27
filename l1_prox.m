function u = l1_prox(x,lambda)
% input: x, lambda
% lambda: magnitude of l1 norm, where the stepsize should be included!
% output: prox_l1(x)
    kernel = abs(x) - lambda*ones(length(x),1);
    logical_mark = ( kernel >=0 );
    u = kernel.*logical_mark.*sign(x);

end

