close all
clc

% Mean, SD, skew, kurtosis
param = [ 1/2, 1/5,  0,   2.5; ...
    1/4, 1/5,  0,   2.5; ...
    1/2, 1/10, 0,   2.5; ...
    1/2, 1/5,  1,   2.5];
titles = {'\mu = 0.5,  \sigma = 0.2, skew = 0',...
    '\mu = 0.25, \sigma = 0.2, skew = 0',...
    '\mu = 0.5,  \sigma = 0.1, skew = 0',...
    '\mu = 0.5,  \sigma = 0.2, skew = 1'};

S = 100;
T = 100;
t=1:1:T;
Q_plus_all = zeros(T,S,4);
Q_minus_all = zeros(T,S,4);
R_all = zeros(T,S,4);
Q_all = zeros(T,S,4);



% Initialize colors
Colors
%% Q^+ vs Q^- and Q. This is row averages, meaning average of Q_i over s=1:4000.

for i = 1:4
    for s = 1:S
        rng(s)
        [Q_plus, Q_minus,rewards] = learning_the_model(0.4, 0.8, 0.7, -0.7, T,param(i,:));
        Q_plus_all(:,s,i) = Q_plus;
        Q_minus_all(:,s,i) = Q_minus;
        rng(s)
        [Q,~,~] = learning_the_model(0.4, 0.8, 0.0, 0.0, T,param(i,:));
        Q_all(:,s,i) = Q;
        R_all(:,s,i) = rewards;
        
    end
end
%% learning
figure(1);
set(gcf,'renderer','Painters')
for i = 1:4
    
    
    subplot(2,2,i);
    
    % Calculate means over simulation runs
    plus = mean(Q_plus_all(:,:,i), 2);
    minus = mean(Q_minus_all(:,:,i), 2);
    Q = mean(Q_all(:,:,i), 2);
    
    
    
    % 25%, average, 75% percentiles of Q^-
    dummy_index = i;
    minus_Q = prctile(Q_minus_all(:,:,dummy_index),[25,75] ,2);
    plus_Q = prctile(Q_plus_all(:,:,dummy_index),[25,75] ,2);
    temp_Q = prctile(Q_all(:,:,dummy_index), [25,75] ,2);
    
    %-----------------------------------------------------------
    % Plotting mean curves and IQR confidence intervals
    
    % Plot mean lines (need to plot first for legend)
    PrettyFig
    hold on
    plot(t,plus,  'Color',clr(1,:),'LineWidth',1.75)
    plot(t,minus, 'Color',clr(2,:),'LineWidth',1.75)
    plot(t,Q,'--','Color',clr(7,:),'LineWidth',1.75)
    
    % Plot IQR for Q+ and Q-
    ShadedErrorBars(t,plus_Q,clr(1,:))
    ShadedErrorBars(t,minus_Q,clr(2,:))
    ShadedErrorBars(t, temp_Q,clr(7,:))
    
    % Re-plot mean lines (to put over shaded regions)
    plot(t,plus,  'Color',clr(1,:),'LineWidth',1.75)
    plot(t,minus, 'Color',clr(2,:),'LineWidth',1.75)
    plot(t,Q,'--','Color',clr(7,:),'LineWidth',1.75)
    % Set axes and their labels
    xlim([0 T]);
    ylim([0,4]);
    if i >= 3
        xlabel( 'Trials' );
    end
    if mod(i,2)==1
        ylabel( 'State-action value' );
    end
    
    % Add title with mean, SD, and skewness
    title(titles{i},'FontWeight','normal')
    
    % Re-size image
    set(gcf,'Position',[10 10 800 600])
    if i==2
        legend({'Q^+','Q^-','Q'}, 'Location', 'Best')
    else
        legend off
    end
    disp([])
end
print('IQRPlot','-dpdf','-r600','-bestfit')
print('IQRPlot','-depsc','-r600','-painters')

%% prepping contour k^+ vs k^- plots for  at t = T.
n = 0.05;
for p = 1:20
    for m = 1:20
        k(p,m,1) = (p-1)*n;
        % second index will encode k_minus
        k(p,m,2) = (m-1)*n;
    end
end
Q_p = zeros(20,20,4);
Q_m = zeros(20,20,4);
for p = 1:20
    for m = 1:20
        for i = 1:4
            temp_plus = zeros(S,1);
            temp_minus = zeros(S,1);
            for s = 1:S
                [plus, minus, ~] = learning_the_model(0.4, 0.8,k(p,m,1), -k(p,m,2),T,param(i,:));
                temp_plus(s) = plus(T,1);
                temp_minus(s) = minus(T,1);
            end
            Q_p(p,m,i) = mean(temp_plus(:));
            Q_m(p,m,i) = mean(temp_minus(:));
        end
    end
end
%% Contours for all 4 reward distributions
figure(2);

spacing_plus = [1.4 1.8 2.2 2.6 3.0 3.4 3.8];
for  i = 1:4
    subplot(2,2,i);
    
    contourf(k(:,:,2), k(:,:,1), (Q_p(:,:,i)+Q_m(:,:,i))/2,'ShowText', 'on');
    pbaspect([1 1 1]);
    title(titles{i},'FontWeight','normal');
    xlabel('k^-');
    ylabel('k^+');
end
sgtitle('Contour of (Q^+(T)+Q^-(T))/2');

figure(3);

spacing_minus = [1.2 1.4 1.6 1.8 2.0];
for  i = 1:4
    subplot(2,2,i);
    
    contourf(k(:,:,2), k(:,:,1), Q_p(:,:,i) - Q_m(:,:,i),'ShowText', 'on');
    pbaspect([1 1 1]);
    title(titles{i},'FontWeight','normal');
    xlabel('k^-');
    ylabel('k^+');
end
sgtitle('Contour of Q^+(T)-Q^-(T)');
%% Contour of (Q^+ + Q^-)/2 and (Q^+-Q^-) at t = T for the first reward dist
figure(4);
subplot(2,1,1);
contourf(k(:,:,2), k(:,:,1), (Q_p(:,:,1) + Q_m(:,:,1))/2, 'ShowText', 'on');
title('(Q^+(T)+Q^-(T))/2', 'FontWeight', 'normal');
xlabel('k^-');
ylabel('k^+');
pbaspect([1 1 1]);

subplot(2,1,2);
contourf(k(:,:,2), k(:,:,1), (Q_p(:,:,1) - Q_m(:,:,1)), 'ShowText', 'on');
title('Q^+(T)-Q^-(T)', 'FontWeight', 'normal');
xlabel('k^-');
ylabel('k^+');
pbaspect([1 1 1]);

sgtitle(titles{1}, 'FontWeight', 'normal');
print('Contour','-dpdf','-r600','-bestfit')
print('Contour','-depsc','-r600','-painters')
%%
% Single simulation

figure(5);
title('Single run');
for i = 1:4
    dummy_index  = i;
    subplot(2,2,dummy_index);
    s = 1:1:S;
    Q_plus = Q_plus_all(:,1,dummy_index);
    Q_minus = Q_minus_all(:,1,dummy_index);
    Q_1 = Q_all(:,1,dummy_index);
    minus_Q_1 = prctile(Q_minus_all(:,1,dummy_index),[25,75] ,1);
    plus_Q_1 = prctile(Q_plus_all(:,1,dummy_index),[25,75] ,1);
    %--------------------------------------------
    % Plotting.
    PrettyFig
    hold on;
    plot(t,Q_plus,  'Color',clr(1,:),'LineWidth',1.75)
    plot(t,Q_minus, 'Color',clr(2,:),'LineWidth',1.75)
    plot(t,Q_1,'--','Color',clr(7,:),'LineWidth',1.75)
    
    
    % Re-plot mean lines (to put over shaded regions)
    plot(t,Q_plus,  'Color',clr(1,:),'LineWidth',1.75);
    plot(t,Q_minus, 'Color',clr(2,:),'LineWidth',1.75);
    plot(t,Q_1, '--', 'Color',clr(7,:), 'LineWidth', 1.75);
    
    % Set axes and their labels
    xlim([0 T]);
    ylim([0 4]);
    if i >= 3
        xlabel( 'Trials' );
    end
    if mod(i,2)==1
        ylabel( 'State-action value' );
    end
    
    % Add title with mean, SD, and skewness
    title(titles{i},'FontWeight','normal')
    
    % Re-size image
    set(gcf,'Position',[10 10 800 600])
    if i==2
        legend({'Q^+','Q^-','Q'}, 'Location', 'Best')
    else
        legend off
    end
    disp([])
    
    print('SinglePlot','-dpdf','-r600','-bestfit')
    print('SinglePlot','-depsc','-r600','-painters')
end


%% modify it so that Q is in there
function [Q_plus, Q_minus,rewards] = learning_the_model(alpha,gamma, k_plus, k_minus, T, v)
Q_plus = zeros(T,1);
Q_minus = zeros(T,1);
rewards = zeros(T,1);
a = v(1);
b = v(2);
c = v(3);
d = v(4);
for t=1:T-1
    %param is a length 4 array.
    %[x,y,z,q] = Beta(a,b,c,d);
    %R = pearsrnd(x,y,z,q);
    R = pearsrnd(a,b,c,d);
    rewards(t,1) = R;
    error_plus = R + (gamma-1)*Q_plus(t,1);
    Q_plus(t+1,1) =Q_plus(t,1) + alpha*pwise(error_plus,k_plus);
    
    error_minus = R + (gamma-1).*Q_minus(t,1);
    Q_minus(t+1,1) =Q_minus(t,1) + alpha*pwise(error_minus,k_minus);
end

end

function f = pwise(error, k)
if (error >= 0)
    f = (1+k)*error;
else
    f = (1-k)*error;
end
end

function[mu,sigma,skew,kurtosis] = Beta(a, b, c, d)
mu = (a*d+b*c)/(a+b);
sigma = (d-c)*sqrt(a*b/(a+b+1))/(a+b);
skew = 2*(b-a)*sqrt((a+b+1)/(a*b))/(a+b+2);
kurtosis = 3 + 6*((a-b)^2*(a+b+1) - a*b*(a+b+2))/(a*b*(a+b+2)*(a+b+3));
end
