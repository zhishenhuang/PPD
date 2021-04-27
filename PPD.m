% Perturbed Gradient Descent
clearvars; close all;

%% parameter setting
dim_grid = [2,5,10,20];
gamma = 1;
L = exp(1);
tau = exp(1);
l = max([2*L,2*gamma,4*L*tau,2*gamma*tau]);
c = 3; % This is where things get sour
rho = 15;
epsilon = 0.01;
delta = 0.1;
Delta_phi = [300, 700, 1500, 3000];
%
lambda = 1e-2; 

maxIter = 1e3;

funVal_perprox = -inf(length(dim_grid),maxIter);
funVal_pergd = -inf(length(dim_grid),maxIter);
funVal_prox = -inf(length(dim_grid),maxIter);
funVal_gd = -inf(length(dim_grid),maxIter);

%% PPD


for dimInd = 1:length(dim_grid)
    
    dim = dim_grid(dimInd);
    xmin = ones(dim,1) * 4*tau;
    [Fun_Phi_min,~] = octopus(xmin, L, gamma, tau);
    fprintf('Current Dimension: %d \n', dim);
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % algo constant setting
    kai = 3*max(log(dim*l*Delta_phi(dimInd)/(c*epsilon^2*delta)),4);
    eta = c/l;
    r = sqrt(c)/kai^2 * epsilon/l;
    r = 0.1;
    g_thres = sqrt(c)/kai^2 * epsilon;
    phi_thres = c/kai^3 * sqrt(epsilon^3/rho);
    t_thres_temp = kai/c^2 * l / sqrt(rho*epsilon);
    
    t_thres_exit = max(50,t_thres_temp);
    
    t_thres = min(40,t_thres_exit);
    t_noise = -t_thres - 1;
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    noise = randn(dim,1);
    x0 = zeros(dim,1);
    x = x0 + noise;
    Fun_Phi_tNoise = inf;

    for t = 1:maxIter

       [Fun_f,Grad_f] = octopus(x, L, gamma, tau);
       Fun_Phi = Fun_f + lambda * norm(x);
       funVal_perprox(dimInd,t) = Fun_Phi - Fun_Phi_min;
       xhat = l1_prox(x - eta*Grad_f,eta*lambda);
       if and(norm(xhat-x)<g_thres , t-t_noise > t_thres)
           xtilde = x;
           t_noise = t;

           [Fun_f_tNoise,~] = octopus(x, L, gamma, tau);
           Fun_Phi_tNoise = Fun_f_tNoise + lambda * norm(x,1);

           randvec = randn(dim,1);
           noise = r*randvec / norm(randvec);
           x = xtilde + noise;
       end
       if and(t-t_noise==t_thres_exit , Fun_Phi - Fun_Phi_tNoise > -phi_thres)
           xstar = xtilde;
           break;
       end
       x = l1_prox(x - eta*Grad_f,eta*lambda);    
       disp(x);
    end
    
end

%% PGD

for dimInd = 1:length(dim_grid)
    dim = dim_grid(dimInd);
    xmin = ones(dim,1) * 4*tau;
    [Fun_Phi_min,~] = octopus(xmin, L, gamma, tau);
    fprintf('Current Dimension: %d \n', dim);
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % algo constant setting
    kai = 3*max(log(dim*l*Delta_phi(dimInd)/(c*epsilon^2*delta)),4);
    eta = c/l;
    r = sqrt(c)/kai^2 * epsilon/l;
    r = 0.1;
    g_thres = sqrt(c)/kai^2 * epsilon;
    phi_thres = c/kai^3 * sqrt(epsilon^3/rho);
    t_thres_temp = kai/c^2 * l / sqrt(rho*epsilon);
    
    t_thres_exit = max(50,t_thres_temp);
%     t_thres = t_thres_temp;
%     t_thres_exit = t_thres_temp;
    t_thres = min(40,t_thres_exit);
    t_noise = -t_thres - 1;
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    noise = randn(dim,1);
    x0 = zeros(dim,1);
    x = x0 + noise;
    Fun_Phi_tNoise = inf;

    for t = 1:maxIter

       [Fun_f,Grad_f] = octopus(x, L, gamma, tau);
       Fun_Phi = Fun_f + lambda * norm(x);
       funVal_pergd(dimInd,t) = Fun_Phi - Fun_Phi_min;
       xhat = x - eta*(Grad_f+sign(x)*lambda);
       if and(norm(Grad_f+sign(x)*lambda)<g_thres , t-t_noise > t_thres)
           xtilde = x;
           t_noise = t;

           [Fun_f_tNoise,~] = octopus(x, L, gamma, tau);
           Fun_Phi_tNoise = Fun_f_tNoise + lambda * norm(x,1);

           randvec = randn(dim,1);
           noise = r*randvec / norm(randvec);
           x = xtilde + noise;
       end
       if and(t-t_noise==t_thres_exit , Fun_Phi - Fun_Phi_tNoise > -phi_thres)
           xstar = xtilde;
           break;
       end
       x = x - eta*(Grad_f+sign(x)*lambda);    
       disp(x);
    end
    
end


%% PD
for dimInd = 1:length(dim_grid)
    dim = dim_grid(dimInd);
    xmin = ones(dim,1) * 4*tau;
    [Fun_Phi_min,~] = octopus(xmin, L, gamma, tau);
    fprintf('Current Dimension: %d \n', dim);
    eta = c/l;
    noise = randn(dim,1);
    x0 = zeros(dim,1);
    x = x0 + noise;
    Fun_Phi_tNoise = inf;

    for t = 1:maxIter

       [Fun_f,Grad_f] = octopus(x, L, gamma, tau);
       Fun_Phi =  Fun_f + lambda * norm(x);
       funVal_prox(dimInd,t) = Fun_Phi - Fun_Phi_min;
       x = l1_prox(x - eta*Grad_f,eta*lambda);    
       disp(x);
    end
    
end

%% GD

for dimInd = 1:length(dim_grid)
    dim = dim_grid(dimInd);
    xmin = ones(dim,1) * 4*tau;
    [Fun_Phi_min,~] = octopus(xmin, L, gamma, tau);
    fprintf('Current dimension: %d \n',dim);
    x0 = zeros(dim,1);
    noise = r*rand(dim,1);
    x = x0 + noise;
    eta = c/l;
    for t = 1:maxIter

        [Fun_f,Grad_f] = octopus(x, L, gamma, tau);
        Fun_Phi = Fun_f + lambda * norm(x);
        funVal_gd(dimInd,t) = Fun_Phi - Fun_Phi_min;
        % find gradient of l1
        Grad_g = sign(x) * lambda;
        x = x - eta * (Grad_f + Grad_g);
        disp(x);

    end

end

%% plot
figure(1); clf;
subplot(2,2,1);
plot(1:1:maxIter,funVal_perprox(1,:),'LineStyle','-','LineWidth',3,'Color','red');
hold on;
plot(1:1:maxIter,funVal_pergd(1,:),'LineStyle','-','LineWidth',3,'Color','yellow');
plot(1:1:maxIter,funVal_prox(1,:),'LineStyle','-.','LineWidth',3,'Color','green');
plot(1:1:maxIter,funVal_gd(1,:),'LineStyle',':','LineWidth',3,'Color','blue');
hold off;
legend('PPD','PGD','PD','GD','location','best','FontSize',25);
xlab = xlabel('Iteration, $d = 2$');
ylab = ylabel('\boldmath$\Phi - \Phi_{\min}$','FontSize',48,'Rotation',90,'VerticalAlignment','middle','HorizontalAlignment','left');
set(xlab, 'Interpreter','latex','FontSize',48);
set(ylab, 'Interpreter','latex');
set(gca,'FontSize',30);

subplot(2,2,2);
plot(1:1:maxIter,funVal_perprox(2,:),'LineStyle','-','LineWidth',3,'Color','red');
hold on;

plot(1:1:maxIter,funVal_pergd(2,:),'LineStyle','-','LineWidth',3,'Color','yellow');
plot(1:1:maxIter,funVal_prox(2,:),'LineStyle','-.','LineWidth',3,'Color','green');
plot(1:1:maxIter,funVal_gd(2,:),'LineStyle',':','LineWidth',3,'Color','blue');
hold off;
legend('PPD','PGD','PD','GD','location','best','FontSize',25);
xlab = xlabel('Iteration, $d = 5$');
ylab = ylabel('\boldmath$\Phi - \Phi_{\min}$','FontSize',48,'Rotation',90,'VerticalAlignment','middle','HorizontalAlignment','left');
set(xlab, 'Interpreter','latex','FontSize',48);
set(ylab, 'Interpreter','latex','FontSize',48);
set(gca,'FontSize',30);

subplot(2,2,3);
plot(1:1:maxIter,funVal_perprox(3,:),'LineStyle','-','LineWidth',3,'Color','red');
hold on;
plot(1:1:maxIter,funVal_pergd(3,:),'LineStyle','-','LineWidth',3,'Color','yellow');
plot(1:1:maxIter,funVal_prox(3,:),'LineStyle','-.','LineWidth',3,'Color','green');
plot(1:1:maxIter,funVal_gd(3,:),'LineStyle',':','LineWidth',3,'Color','blue');
hold off;
legend('PPD','PGD','PD','GD','location','best','FontSize',25);
xlab = xlabel('Iteration, $d = 10$');
ylab = ylabel('\boldmath$\Phi - \Phi_{\min}$','FontSize',48,'Rotation',90,'VerticalAlignment','middle','HorizontalAlignment','left');
set(xlab, 'Interpreter','latex','FontSize',48);
set(ylab, 'Interpreter','latex','FontSize',48);
set(gca,'FontSize',30);

subplot(2,2,4);
plot(1:1:maxIter,funVal_perprox(4,:),'LineStyle','-','LineWidth',3,'Color','red');
hold on;
plot(1:1:maxIter,funVal_pergd(4,:),'LineStyle','-','LineWidth',3,'Color','yellow');
plot(1:1:maxIter,funVal_prox(4,:),'LineStyle','-.','LineWidth',3,'Color','green');
plot(1:1:maxIter,funVal_gd(4,:),'LineStyle',':','LineWidth',3,'Color','blue');
hold off;
legend('PPD','PGD','PD','GD','location','best','FontSize',25);
xlab = xlabel('Iteration, $d = 20$');
ylab = ylabel('\boldmath$\Phi - \Phi_{\min}$','FontSize',48,'Rotation',90,'VerticalAlignment','middle','HorizontalAlignment','left');
set(xlab, 'Interpreter','latex','FontSize',48);
set(ylab, 'Interpreter','latex','FontSize',48);
set(gca,'FontSize',30);

%%
figure(2); clf;
plot(1:1:maxIter,funVal_perprox(2,:),'LineStyle','-','LineWidth',3,'Color','red');
hold on;
plot(1:1:maxIter,funVal_gd(2,:),'LineStyle',':','LineWidth',1,'Color','green');
hold off; 
legend('Prox','GD','location','best');
xlabel('Iteration');
ylab = ylabel('$\Phi$');
set(ylab, 'Interpreter','latex');


%% 2D function plot test
x = -6*tau:0.1:6*tau;
y = -6*tau:0.1:6*tau;

XX = zeros(length(y),length(x));
YY = zeros(length(y),length(x));
for ind = 1:length(y)
    XX(ind,:) = x;
end
for ind = 1:length(x)
    YY(:,ind) = y';
end
ZZ = zeros(length(y),length(x));

for indX = 1:length(x)
    for indY = 1:length(y)
        vector = [XX(indY,indX) YY(indY,indX)]; 
%         [ZZ(indY,indX),~] = octopus(vector,L,gamma,tau);
        [temp,~] = octopus(vector,L,gamma,tau);
        ZZ(indY,indX) = temp + lambda * norm(vector, 1);
    end
end

figure(1); clf;
meshc(XX,YY,ZZ);
axis tight
xlab = xlabel('\boldmath$x$'); 
set(xlab,'Interpreter','latex','FontSize',48);
ylab = ylabel('\boldmath$y$'); 
set(ylab,'Interpreter','latex','FontSize',48);
zlab = zlabel('\boldmath$z$'); 
set(zlab,'Interpreter','latex','FontSize',48);
set(gca,'FontSize',36);

