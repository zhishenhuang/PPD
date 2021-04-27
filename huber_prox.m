function u = huber_prox(x,lambda,omega)
% lambda: magnitude of l1 term
% omega: huber parameter

    u = zeros(length(x),1);
    for ind=1:length(x)
        if abs(x(ind))<= lambda+1/omega
            u(ind) = 1/(1+lambda*omega) * x(ind);
        elseif x(ind)>lambda+1/omega
            u(ind) = x(ind)-lambda;
        else
            u(ind) = x(ind)+lambda;
        end
    end

end

