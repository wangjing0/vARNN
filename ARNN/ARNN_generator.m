function arnn = ARNN_generator(arnn)

if isfield(arnn,'Nseq'); Nseq=arnn.Nseq;
else; Nseq = 1000; end

if isfield(arnn,'Nrepeat'); Nrepeat=arnn.Nrepeat;
else; Nrepeat = 100; end

if isfield(arnn,'Nsession'); Nsession=arnn.Nsession;
else; Nsession = 10; end

if isfield(arnn,'AR_coeff'); AR_coeff=arnn.AR_coeff;
else; AR_coeff =[0.13 0.11	0.09  0.08	0.07	0.06	0.06	0.05	-0	-0.03];
end; p = length(AR_coeff); % order of ar process

if isfield(arnn,'alpha_beta'); alpha_beta=arnn.alpha_beta;
else;alpha_beta = .3; end
alpha = alpha_beta(1); beta = alpha_beta(2);

if isfield(arnn,'noise'); noise=arnn.noise;
else; noise = 0.15; end

if isfield(arnn,'k_reward'); k_reward0=arnn.k_reward;
else; k_reward0 = 0.5*noise; end

if isfield(arnn,'dk_reward'); dk_reward=arnn.dk_reward;
else; dk_reward = 0.01; end

%simulate AR
Y = [];
R = [];
for nnn=1:Nsession
    disp(['Session ',num2str(nnn),'/',num2str(Nsession)])
    y = nan(Nseq,Nrepeat);
    reward = nan(Nseq,Nrepeat);
    k_reward = k_reward0.*ones(1,Nrepeat);
    for i = 1:Nseq
        if i<=p
            y(i,:) = noise*randn(1,Nrepeat);
            reward(i,:) = (abs(y(i,:))<k_reward);%.* (1 - abs(y(i,:)./k_reward));
        else
            clear y_temp reward_temp
            y_temp = y(i-p:i-1,:);
            r_temp = reward(i-p:i-1,:);
            r_noise = (beta*(1-r_temp)+ alpha.*(r_temp)); % reward modulates noise variance, if r = 1, variance = alpha,  r = 0, variance = beta
            e_noise = noise*randn(1,Nrepeat);
            y(i,:) = AR_coeff(end:-1:1) * (y_temp.*(1 + sqrt(r_noise).*randn(size(y_temp)))) +  e_noise;
            reward(i,:) = (abs(y(i,:))<k_reward);%.* (1 - abs(y(i,:)./k_reward));
            k_reward = k_reward + dk_reward*( 0.5 - (reward(i,:)>0)); % on staircase
        end
    end
    y(1:p,:) = nan; % nan padding at the initial samples
    reward(1:p,:) = nan;
    Y = [Y,reshape(y,[],1)];
    R = [R,reshape(reward,[],1)];
%   %plot coefficients, variability
%     if PLOT
%         klow = 1.3; khigh= 5;
%         N_kbins = 20;
%         kreward = [-1*2.^(linspace(-klow,-khigh,ceil(N_kbins/2))), 0, fliplr(2.^(linspace(-klow,-khigh,ceil(N_kbins/2))))];
%         kreward =zeros(N_kbins,1);
%         if nnn==1
%             %  figure;
%             subplot(311)
%             bar([1:p]+alpha_beta(2)*.5,[AR_coeff],'BarWidth',.5,'EdgeColor','none','FaceColor',alpha_beta(2)*[0,0,1]); hold on;
%             % makeaxis('y_label','AR coefficients')
%         end
%         subplot(312);  ylim([noise 2*noise]);
%         z = reshape(y,[],1);
%         z_1= [nan;z(1:end-1)];
%         [mTp_stdTp] = mu_std_Tp(z,z_1,kreward,zeros(size(z)),0,1,0,0);
%         %plot(mTp_stdTp(:,1),mTp_stdTp(:,2),'k-o','LineWidth',2); hold on;
%         plot(mTp_stdTp(:,1),mTp_stdTp(:,3),'.','Color',alpha_beta(2)*[0,0,1]);  drawnow; hold on
%         if nnn==Nsession
%             z = reshape(Y,[],1);z_1= [nan;z(1:end-1)];
%             [mTp_stdTp] = mu_std_Tp(z,z_1,kreward,zeros(size(z)),0,1,0,0);
%             subplot(312);
%             plot(mTp_stdTp(:,1),mTp_stdTp(:,3),'-','LineWidth', 2 ,'Color',alpha_beta(2)*[0,0,1]);  drawnow; hold on
%             
%             subplot(313)
%             maxNlags = 40;
%             [pc,~,cf] = parcorr(z,maxNlags,0,2.0);
%             pc = pc(2:end);
%             Nlags = 1:maxNlags;
%             plot(Nlags,ones(size(Nlags)).*cf(1),'k:');hold on;
%             plot(Nlags,-ones(size(Nlags)).*cf(1),'k:');hold on;
%             plot(Nlags,pc,'-','color','k'); hold on; drawnow;
%         end
%         % if nnn==Nsession; makeaxis('x_label','y(t-1)','y_label','\sigma(y(t))');  end
%     end
end
arnn.Y = Y;
arnn.Reward = R;
return