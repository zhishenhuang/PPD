function hess = huber_hessian(x,lambda,omega)
% input: x, lambda, omega
    diag_val = lambda*omega*ones(length(x),1).*(abs(x)<=1/omega);
    hess = diag(diag_val);
end

