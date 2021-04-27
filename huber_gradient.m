function grad = huber_gradient(x, lambda,omega)
% input: x, lambda, omega
% x: input vector
% lambda: magnitude of l1 term
% omega: huber parameter

    grad = zeros(length(x),1);
    for ind = 1:length(x)
        if abs(x(ind))<=1/omega
            grad(ind) = lambda*omega*x(ind);
        else
            grad(ind) = lambda*sign(x(ind));
        end
    end

end