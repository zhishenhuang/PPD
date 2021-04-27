function [fun,grad] = octopus(x,L,gamma,tau)
% Input: ( x, L, gamma, tau)
% Assume that larger components in x always bear large indices
% L, gamma, and tau are octopus function parameters as defined in Du et al.
% GD can take exponential time to escaple saddle point (arXiv:1705.10412)
% Output: [fun, grad]
% Need to require that xAbs < 6*tau

wiggle = 1e-5;
dim = length(x);
% nu = (26*L+2*gamma)/3*tau^2 + (-5*L+3*gamma)/2*tau^3;
nu = 13/6*gamma*tau^2 + 37/6*L*tau^2;
xAbs = abs(x);

grad = zeros(dim,1);

%% function value
recThreshold = (xAbs <= 2*tau);
ind = find(recThreshold,1); 
% This "ind" is the ind of x_i


if any( xAbs(ind+1:end)> tau ) % Outside domain, disgard
    fun = inf;
    grad = inf(dim,1);
    return

elseif isempty(ind) % all x_i greater than 2*tau
    fun = L*sum((xAbs-4*tau).^2) - dim * nu;
    % Equation (10)
    
else % Interior Point: for all j>=ind, x_j<tau
    
    if ind < dim % corresponding to the case "i<d"
        if xAbs(ind)<=tau
            fun = L * sum( (xAbs(1:ind-1)-4*tau).^2 ) - gamma * xAbs(ind)^2 +...
                L* sum( xAbs(ind+1:dim).^2 ) - (ind-1)*nu; % Equantion (6)
        else
            fun = L * sum( (xAbs(1:ind-1)-4*tau).^2 ) +...
                g( xAbs, ind, L, gamma, tau ) + ...
                L* sum( xAbs(ind+2:dim).^2 ) - (ind-1)*nu; % Equation (7)
        end
    else % corresponding to the case "i=d"
        if xAbs(dim)<=tau
            fun = L * sum((xAbs(1:dim-1)-4*tau).^2)  - gamma*xAbs(dim)^2 - (dim-1)*nu;
            % Equation (8)
        elseif (xAbs(dim)<=2*tau)&&(xAbs(dim)>tau)
            fun = L * sum((xAbs(1:dim-1)-4*tau).^2) + g1(xAbs(dim),L,gamma,tau,1) -...
                (dim-1)*nu;
            % Equation (9)
        end
    end
    
end

%% gradient value

sign_rec_Ind = intersect(1:ind-1,find(x>=0));
sign_rec_IndC = setdiff(1:ind-1,sign_rec_Ind);

if isempty(ind)
    grad(x>=0) = 2*L*(xAbs(x>=0)-4*tau);
    grad(x<0) = -2*L*(xAbs(x<0)-4*tau);
    
else
    if ind < dim % i<d case
        if xAbs(ind)<=tau           
            grad( sign_rec_Ind ) = 2*L*(xAbs(sign_rec_Ind) - 4*tau);
            grad( sign_rec_IndC ) = -2*L*(xAbs(sign_rec_IndC) - 4*tau);
            grad(ind) = -2*gamma*x(ind);
            grad(ind+1:dim) = 2*L*x(ind+1:dim);
        else
            grad( sign_rec_Ind ) = 2*L*(xAbs(sign_rec_Ind) - 4*tau);
            grad( sign_rec_IndC ) = -2*L*(xAbs(sign_rec_IndC) - 4*tau);
            [~,tempGrad] = g(xAbs,ind,L,gamma,tau);        
            grad(ind:ind+1) = reshape(tempGrad(ind:ind+1),[length(tempGrad(ind:ind+1)),1]) .* reshape(sign(x(ind:ind+1)),[length(sign(x(ind:ind+1))),1]);
            grad(ind+2:dim) = 2*L*x(ind+2:dim);
        end
    elseif ind == dim     % i=d case
        if xAbs(dim)<=tau
            grad( sign_rec_Ind ) = 2*L*(xAbs(sign_rec_Ind) - 4*tau);
            grad( sign_rec_IndC ) = -2*L*(xAbs(sign_rec_IndC) - 4*tau);
            grad(dim) = -2*gamma*x(dim);
        elseif xAbs(dim)<=2*tau
            grad( sign_rec_Ind ) = 2*L*(xAbs(sign_rec_Ind) - 4*tau);
            grad( sign_rec_IndC ) = -2*L*(xAbs(sign_rec_IndC) - 4*tau);
            grad(dim) = g1(xAbs(dim),L,gamma,tau,2) * sign(x(dim));
        end
    end
       
end


end % function end