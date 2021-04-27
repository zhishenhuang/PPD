%% Gradient Descent vs Newton's method for non-convex problems
% We look at a 2D problem with local minima, local maxima, and a saddle point.
% 
% Stephen Becker, June 2018, Cambridge/CCIMI Optimisation short-course
%
% *Newton's method*, on the other hand, can converge to a saddle point or 
% even a local max! This is quite undesirable. Why does it do this? It is trying 
% to solve the stationarity equations (which treat all local min, local max, and 
% saddle points equally), and it finds whichever stationary point is closest. 
% [For real applications, no one uses plain Newton without any kind of safe guard, 
% or at least they ought not to, but there is still a problem with saddle points]
% 
% See "<https://arxiv.org/abs/1406.2572 Identifying and attacking the saddle 
% point problem in high-dimensional non-convex optimization>" by Yann Dauphin, 
% Razvan Pascanu, Caglar Gulcehre, Kyunghyun Cho, Surya Ganguli, Yoshua Bengio 
% (NIPS 2014)
% 
% On the other hand, at least Newton's method converges to these undesirable 
% points very quickly...
% 
% 
% 
% Some code to setup our example:
% A function "f" with a saddle point
clear all; clc;
cd /Users/leonardohuang/Desktop/Octopus/
f  = @(X,Y) 1/2*(X.^2 - Y.^2);
% make this only a local function: multiply by Gaussian "g"
% (otherwise, it has no minimizers)
sigma2  = 5;
g       = @(X,Y) exp( -(X.^2+Y.^2)/sigma2 );
lambda  = 1e-2; %.1;  if lambda > 1e-2, the origin is a local min!
omega   = 100;
phi     = @(X,Y) norm(X) + norm(Y);

Huber   = @(X,Y)  huber([X,Y],omega);

xs = 0; ys = xs;
shift_huber = [xs,ys];

% The function we are interested in is h = f*g + lambda*||x||_1
% h       = @(X,Y) f(X,Y).*g(X,Y) + lambda * Huber(X,Y);
h       = @(X,Y) f(X,Y).*g(X,Y) + lambda * Huber(X-xs,Y-ys);
% Find partial derivatives (w.r.t. x and then y)
% !Gradient of Huber is added in Grad definition!
h_x  = @(X,Y) X.*g(X,Y) - 1/sigma2*(X.^2-Y.^2).*X.*g(X,Y) ;
h_y  = @(X,Y) -Y.*g(X,Y) - 1/sigma2*(X.^2-Y.^2).*Y.*g(X,Y);
% Make Hessian
h_xx = @(X,Y) 2*X.^4 + sigma2*(Y.^2+sigma2)-X.^2*(2*Y.^2+5*sigma2);
h_xy = @(X,Y) 2*X.*Y.*(X.^2-Y.^2);
h_yx = @(X,Y) h_xy(X,Y);
h_yy = @(X,Y) -2*Y.^4 +   5   *Y.^2*sigma2-sigma2^2+X.^2.*( 2*Y.^2-sigma2);
hess = @(X,Y) g(X,Y)/(sigma2^2).*[ h_xx(X,Y), h_xy(X,Y); h_yx(X,Y), h_yy(X,Y) ];
% Make versions of the functions that take in xvec=[x;y]
F       = @(xvec) h(xvec(1),xvec(2));
Grad    = @(xvec) [h_x( xvec(1), xvec(2) ); h_y( xvec(1), xvec(2) ) ] +...
            huber_gradient(xvec-shift_huber,lambda,omega); 
Grad_fOnly    = @(xvec) [h_x( xvec(1), xvec(2) ); h_y( xvec(1), xvec(2) ) ];
Hess    = @(xvec) hess( xvec(1), xvec(2) ) + huber_hessian(xvec - reshape(shift_huber,size(xvec)),lambda,omega);
fprintf('Eigenvalues of Hessian are: ');
evals = eig( Hess( [0;0] ) );
for i = 1:2, fprintf('%g  ', evals(i) ); end; fprintf('\n');

[X,Y]   = meshgrid( linspace( -5,5,200),linspace(-5,5,200) );
Z       = zeros(size(X));
for indx = 1:size(X,2)
    for indy = 1:size(X,1)
        Z(indy,indx) = h(X(indy,indx),Y(indy,indx));
    end 
end

[approxMin,minLocation] = min(Z(:));
xMin = X( minLocation );
yMin = Y( minLocation );

%% Plot
figure(1); clf;
meshc( X, Y, Z); 
xlab=xlabel('\textbf{x}'); ylab=ylabel('\textbf{y}'); zlab=zlabel('\textbf{\textrm{Function value}}');
set(gca,'FontSize',30)
set(xlab,'Interpreter','latex','FontSize',48);
set(ylab,'Interpreter','latex','FontSize',48);
set(zlab,'Interpreter','latex','FontSize',48);




%% Run both methods, for two different starting points
%%
maxIter = 1e3;
funcVal_prox = zeros(2,maxIter+1);
funcVal_gd = zeros(2,maxIter+1);
for trial = 1:2
    switch trial
        case 1
            x0      = [.3; -0.01 ]; xRef=[.447213595499958;0]; % Newton goes to local max!
        case 2
%             x0      = [.1; -0.01 ]; xRef = [0;0];
            x0      = [2; 0.01 ]; xRef = [0;0];
    end
    
    figure(trial); clf;
    meshc( X, Y, Z );
    xlabel('x'); ylabel('y'); zlabel('Function value');
    hold all
    
    % Gradient Descent
    x       = x0;   
    t       = 1/2; % stepsize
    funcVal_gd(trial,1) = F(x0);
    for k = 1:maxIter
        handles = plot3( x(1), x(2), F(x)+.0001, 'ro','markersize',6 );
        handles.MarkerFaceColor=handles.Color;        
        x   = x - t*Grad(x);
        funcVal_gd(trial,k+1) = F(x);
    end
    
    % Proximal Descent
    x       = x0;
    t       = 1/2;  % stepsize
    funcVal_prox(trial,1) = F(x);
    for k = 1:maxIter
        handles = plot3( x(1), x(2), F(x)+.0001, 'b*','markersize',6 );
        handles.MarkerFaceColor=handles.Color;   
        x   = huber_prox(x-t*Grad_fOnly(x)-shift_huber,t*lambda,omega)+shift_huber; 
        funcVal_prox(trial,k+1) = F(x);
    end
    
    disp('One Iteration Finished.!')
       
    % Newton's method
%     x       = x0;
%     fprintf('Newton''s method:\nIter %d, distance(x,stationary point) is %.2e\n', 0, norm(x) );
%     
%     for k = 1:5
%         handlesN = plot3( x(1), x(2), F(x)+.0001, 'bs','markersize',6 );
%         handlesN.MarkerFaceColor=handlesN.Color;
%         
%         H   = Hess(x);
%         x   = x - (H\Grad(x));
%         fprintf('Iter %d, distance(x,stationary point) is %.2e\n', k, norm(x-xRef) );
%     end
%     
%     switch trial
%         case 1
%             title('Newton''s method converges to local max');
%         case 2
%             title('Newton''s method converges to saddle point');
%     end
%     legend([handles,handlesN],'Gradient Descent','Newton''s Method','location','best');
%       legend(handles,'Gradient Descent','location','best');
end
disp('done')

%%

figure(3); clf;
iter = 1:1:maxIter+1;
hold on;
plot(iter,funcVal_gd(1,:),'LineWidth',2,'Color','r');
plot(iter,funcVal_prox(1,:),'LineWidth',2,'Color','g');
hold off;
legend('GD','Prox')
xlabel('Iteration')
ylabel('Function Value')

figure(4); clf;
% yyaxis left
hold all;
plot(iter,funcVal_gd(2,:),'LineWidth',2,'Color','r');
% yyaxis right
plot(iter,funcVal_prox(2,:),'LineWidth',2,'Color','g');
legend('GD','Prox')
xlabel('Iteration')
ylabel('Function Value')

%%
% Success definition 1: F(x_T)-F(xStar)<thres
% Success definition 2: (F(x_T)-F(xStar))/F(xStar)<thres_%
% find max
maxIter = 1e3;
% x0 = [0,-2];
x0 = [xMin, yMin];
x       = x0;
t       = 1/5; % stepsize
thres = 1e-4;
for k = 1:5*1e2
    x   = x - t*Grad(x);
end
xStar = x; % min location
FStar = F(xStar); % min value

if abs( FStar - approxMin ) > 1e-1
    disp('Does not agree with grid');
end


% stepsize_grid = 10.^(linspace(-3,1,200));
stepsize_grid = 10.^(linspace(-4,1,250));
noiseRepeat = 50;
succ_step_gd = zeros(length(stepsize_grid),1);
succ_step_prox = zeros(length(stepsize_grid),1);

count = 1;
trial = 2;
while count<noiseRepeat
    fprintf('\n\n\n Current Noise Trial: % d out of %d \n\n\n',count,noiseRepeat);
    switch trial
        case 1
            x0      = [.3; -0.01 ]; xRef=[.447213595499958;0]; % Newton goes to local max!
        case 2
            %             x0      = [.1; -0.01 ]; xRef = [0;0];
            x0      = [2; 0.01 ]; xRef = [0;0];
    end
    polangle = 2*pi* rand(1);
    r = norm(x0)*1/8;
    noise = [r*cos(polangle),r*sin(polangle)];
    for ind_step = 1:length(stepsize_grid)
        fprintf('Stepsize being tested:  %f \n',stepsize_grid(ind_step));
        % Gradient Descent
        x       = x0+noise;
        t       = stepsize_grid(ind_step); % stepsize
        for k = 1:maxIter
            x   = x - t*Grad(x);
        end
        if abs(F(x) - F(xStar))<thres
            succ_step_gd(ind_step) = succ_step_gd(ind_step) + 1; 
        end
        
        % Proximal Descent
        x       = x0+noise;
        t       = stepsize_grid(ind_step);
        funcVal_prox(trial,1) = F(x);
        for k = 1:maxIter
            x   = huber_prox(x-t*Grad_fOnly(x)-shift_huber,t*lambda,omega)+shift_huber;
        end
        if abs(F(x) - F(xStar))<thres
            succ_step_prox(ind_step) = succ_step_prox(ind_step) + 1; 
        end
    end
    fprintf('\n\n');
    
    count=count+1;
end
succ_step_gd = succ_step_gd / noiseRepeat;
succ_step_prox = succ_step_prox / noiseRepeat;


%%
plotrange = 103:250;
figure(5); clf;
semilogx(stepsize_grid(plotrange),succ_step_gd(plotrange),'LineStyle',':','LineWidth',5,'Color','r');
hold on;
semilogx(stepsize_grid(plotrange),succ_step_prox(plotrange),'LineStyle','-','LineWidth',2,'Color','b');
legend1 = legend('GD','Prox');
xlab = xlabel('\textbf{Stepsize}');
ylab = ylabel('\textbf{Percentage of success}');
% yticks = [get(gca,'ytick')]'; % There is a transpose operation here.
% percentsy = repmat('%', length(yticks),1);  %  equal to the size
% yticklabel = [num2str(yticks*100) percentsy]; % concatenates the tick labels 
% set(gca,'yticklabel',yticklabel) 
set(gca,'fontweight','bold','fontsize',36);
set(legend1,'FontSize',36,'location','best','Interpreter','latex');
set(ylab,'Interpreter','latex','FontSize',48);
set(xlab,'Interpreter','latex','FontSize',48);

xlim([1e-2 1e1])



