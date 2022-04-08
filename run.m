%% ARNN and vARNN time series simulation
clc; clear;  close all;
rng(0);
global arnn
alpha = 0.5^2;
alpha_beta = alpha.*[1e-6, 1e-6; 1, 3];% case1: vanilla AR ; case2: reward modulates variance vAR
W = 0.25*exp(-linspace(0,19,20)/3);
for aaa =1:size(alpha_beta,1)
    arnn = struct('Nseq',1e3,...
        'Nrepeat',10,...
        'Nsession',10,...
        'AR_coeff',W,...
        'alpha_beta',alpha_beta(aaa,:), ...
        'noise',.1,...
        'k_reward',0.05,...
        'dk_reward',0.01);
    arnn = ARNN_generator(arnn);
end
%% search optimal parameters of ARNN, with regularization
arnn.valid_p = 0.2; % percentage of data used for validation
arnn.Validation = rand(size(arnn.Y))<=arnn.valid_p;
arnn.Regularizer.sparsity = .5;
arnn.Regularizer.c1 = 3;
arnn.Regularizer.c2 = 3;

opts = optimset('display','iter-detailed ', 'PlotFcns',{@optimplotx,...
                @optimplotfval},'MaxFunEvals',1e3,'TolX',1e-4);
opts_ = optimset('display','off','MaxFunEvals',1e3,'TolX',1e-5);
PLOT = 0;
Nsession = arnn.Nsession;
W_init_ = [arnn.AR_coeff, arnn.alpha_beta(1)./arnn.alpha_beta(2)]; % true parameters
W_opt = nan(Nsession, length(W_init_));
W_opt_ar = nan(Nsession, length(W_init_));
W_init = repmat(W_init_,Nsession,1).*abs((1+ 0.5*randn(size(W_opt)))); % alpha does not have a nice convex profile
Error =  nan(Nsession,4,2);
for iii=1:Nsession
    disp(['========= Optimizing session #', num2str(iii), ' ========='])
    % optimization
    OPTIM = 1;
    W_opt(iii,:) = fminsearch(@(W)ARNN_estimator(W,'wMSE','train',PLOT,iii,OPTIM),W_init(iii,:),opts_);% find optimal parameters for vARNN
    W_opt_ar(iii,:) = fminsearch(@(W)ARNN_estimator(W,'MSE','train',PLOT,iii,OPTIM),W_init(iii,:),opts_);% optimal for AR model
    % evaluation
    OPTIM = 0;
    Error(iii,1:3,1) = [ARNN_estimator(W_opt_ar(iii,:),'MSE','test',PLOT,iii,OPTIM),...% AR optimized parameters, on test data
                        ...% ARNN_estimator(W_init(iii,:),'MSE','test',PLOT,iii),... % inital parameters, on test data
                        nanmean(ARNN_estimator(W_opt(iii,:),'rwMSE','test',PLOT,iii,OPTIM)),...%% VARNN, loss from randomly weighted the samples, as the null hypthesis
                        ARNN_estimator(W_opt(iii,:),'wMSE','test',PLOT,iii,OPTIM)]; % VARNN, optimal parameters, weighted loss, on test data
    Error(iii,2,2) = nanstd(ARNN_estimator(W_opt(iii,:),'rwMSE','test',PLOT,iii,OPTIM)); %
end
%% before and after optimization, the coefficients is closer to the ground truth??
error_init = sum((W_init - W_init_).^2,2);
error_opt = sum((W_opt - W_init_).^2,2);
figure; hold on
plot([1 2], [error_init error_opt],'k-');
boxplot([error_init error_opt]);
[~,p] = ttest(error_init, error_opt,'Tail','right');
title(['Found the true paras? pairwise ttest p =', num2str(p,3)])
xticks([1 2])
ax = gca;
ax.TickLabelInterpreter = 'tex';
ax.XTickLabel= {'|\theta(init) - \theta(true)|', '|\theta(opt) - \theta(true)|'};
%% LSTM(GRU) model optimization
% dataset
rng(0);
PLOT =1;
[LSTMnet,Xinit_] = LSTM_trainer(arnn.Y,arnn.Reward,PLOT);
%% simulate with trained LSTM
% trained LSTM self generates TS , does it have u-shape variability curve?
Xinit = repmat(Xinit_,arnn.Nrepeat,1);
[Yout,Reward] = LSTM_generator(LSTMnet,Xinit,arnn.Nseq,arnn.k_reward,arnn.dk_reward,arnn.noise);
Nsession = length(Xinit);
figure;
subplot(311);
plot(Yout');
subplot(312);hold on;
kreward =zeros(20,1);
for nnn=randperm(Nsession,10)%1:Nsession
    z = reshape(Yout(nnn,:)',[],1);
    z_1= [nan;z(1:end-1)];
    [mTp_stdTp] = mu_std_Tp(z,z_1,kreward,zeros(size(z)),0,1,0,0);
    plot(mTp_stdTp(:,1),mTp_stdTp(:,3),'.');  drawnow;
end
z = reshape([nan(size(Yout,1),1),Yout]',[],1);z_1= [nan;z(1:end-1)];
[mTp_stdTp] = mu_std_Tp(z,z_1,kreward,zeros(size(z)),0,1,0,0);
plot(mTp_stdTp(:,1),mTp_stdTp(:,3),'-','LineWidth', 2 ,'Color','r');  drawnow; hold on
% acf
maxNlags = 40;
z = reshape(Yout',[],1);
[pc,~,cf] = parcorr(z,maxNlags,0,2.0);
Nlags = 1:maxNlags;
subplot(313);
plot(Nlags,ones(size(Nlags)).*cf(1),'k:');hold on;
plot(Nlags,-ones(size(Nlags)).*cf(1),'k:');hold on;
plot(Nlags,pc(2:end),'-','color','k');
ylim([-.1 .5]);hold on; drawnow;
%% test, error of LSTM
LSTMloss = nan(arnn.Nsession,1);
Nsession = arnn.Nsession;
maxNlags = 40;
PC = nan(Nsession,maxNlags+1);

for sss=1:Nsession
    [X,Y,~,~,indexTrain] = getDataSetLSTM(arnn.Y(:,sss),arnn.Reward(:,sss),arnn.tau,arnn.d,'prediction_steps',1,'split_ratio',1);
    Y = cell2mat(Y);
    Y_pred = predict(LSTMnet,X);
    LSTMloss(sss) = (nanmean((Y - Y_pred).^2))/nanvar(Y); % explained variance/total variance
    [pc,~,cf] = parcorr(Y_pred,maxNlags,0,2.0);
    PC(sss,:) = pc;
end
Error(:,4,1)=LSTMloss;
figure;
subplot(311);
plot(Y(1:5e2),'k.','DisplayName','true'); hold on
plot(Y_pred(1:5e2),'g-','DisplayName','GRU prediction');
legend
subplot(312);
plot(Y,Y_pred,'k.'); hold on
plot([-1 1],[-1 1],'k--');
title(['GRU model loss = ', num2str(nanmean(LSTMloss),3)]);
subplot(313)
pc = nanmean(PC);
Nlags = 1:maxNlags;
plot(Nlags,pc(2:end),'.-','color','k'); hold on; drawnow;
%% plot result of  all 4 models
figure('Position',[0 0 500 700]);
subplot(211)
bar(W_init_,'k','EdgeColor','none','BarWidth',.4,'FaceAlpha',.7); hold on;
for p=1:length(W_init_)
    subplot(211)
    plot([p-.7 p], [W_init(:,p) W_opt(:,p)],'k-'); hold on
    plot([p-.7], [W_init(:,p) ],'k.'); hold on
    plot([p], [W_opt(:,p) ],'b.'); hold on
end
subplot(211); ylim([ -.1 2.5*max(W_init_)]);
%makeaxis('y_label','Parameters','x_label','before/after training')
subplot(212);
Ngroup = size(Error,2);
for i=1:size(Error,1)
    jitter = .03*randn(1);
    plot([1:Ngroup]+jitter,Error(i,:,1),'k-','LineWidth',.5); hold on
    plot(1+jitter,Error(i,1,1),'k.'); hold on
    plot(2+jitter,Error(i,2,1),'k.'); hold on
    errorbar(Ngroup-2+jitter,Error(i,Ngroup-2,1),Error(i,Ngroup-2,2),'k-','LineWidth',1,'capsize',0); hold on
    plot(Ngroup-1+jitter,Error(i,Ngroup-1,1),'b.'); hold on
    plot(Ngroup+jitter,Error(i,Ngroup,1),'k.');
end
% 1,2,3,4 = AR MSE, rwMSE vAR, wMSE vAR, GRU
[~,p1] = ttest(Error(:,3,1), Error(:,1,1),'Tail','left');
[~,p2] = ttest(Error(:,3,1), Error(:,2,1),'Tail','left');
[~,p4] = ttest(Error(:,3,1), Error(:,4,1),'Tail','left');
title(['p = ', num2str([p1,p2,p4],3)])
ylim([ 0.3 .75])
xlim([0 Ngroup+1])
xticks([1:Ngroup])
xticklabels({'\theta^*, MSE AR', '\theta^*, rwMSE,VARNN','\theta^*, wMSE,VARNN','\theta^*, MSE GRU'});
%makeaxis('y_label','Unexplained variance');
%% forecasting, example session
OPTIM = 0;
PLOT=1;
ARNN_estimator(W_opt(1,:),'wMSE','test',PLOT,0,OPTIM);
%%  behavioral data fitting to VARNN and GRU
clc; clear; close all;
load('./data/CSG_All_humanSubjects.mat')
figure;
bin = linspace(-1,1,100)';
e = (Tp-TS)./TS;
ct0= histc(e(Reward==0), bin);
ct1= histc(e(Reward==1), bin);
bar(bin,[ ct1', ct0'],'stacked'); hold on
xlim([-1 1]); makeaxis();
%%
maxNlags = 20;
Nsessions = length(Xcut);
opts = optimset('display','iter-detailed ', 'PlotFcns',{@optimplotx,...
    @optimplotfval},'MaxFunEvals',1e3,'TolX',1e-3,'TolFun',1e-3);
opts_ = optimset('display','off','MaxFunEvals',1e3,'TolX',1e-3,'TolFun',1e-3); % options for optimization
PLOT = 0;
ARNN=repmat({struct()},Nsessions,2);
Loss=repmat({[]},Nsessions,2);% eye, hand
global arnn
for nnn=1:Nsessions
    clear indx tp reward hand long
    if nnn==1
        indx = 1:Xcut(nnn);
    else
        indx = (Xcut(nnn-1)+1):Xcut(nnn);
    end
    tp = (Tp(indx) - TS(indx))./TS(indx); % normalize
    reward = Reward(indx);
    hand = H(indx);
    long = L(indx);
    for h=[0,1]
        clear indy y r
        indy = (hand==h ) ;
        % if nansum(indy) >= 20*maxNlags % only if enough data points for pValue=0.05
        y = tp(indy);
        r = reward(indy);
        try
            y = y - nanmean(y); % zero mean
            arnn = setup_arnn(y,r,maxNlags);
            W_init = [arnn.parcor, .3/1.0];
            [W_opt,fval,exitflag] = fminsearch(@(W)ARNN_estimator(W,'wMSE','train',PLOT,0,1),W_init,opts_);
            arnn.AR_coeff = W_opt(1:end-1);
            arnn.alpha_beta = abs(W_opt(end));
            l0= ARNN_estimator(W_opt,'MSE','test',PLOT,0,0);% loss with null distributed weights
            l1= ARNN_estimator(W_opt,'wMSE','test',PLOT,0,0);% loss of the optimal weighted
            
            arnn.exitflag = exitflag;
            arnn.l0 = l0;%[nanmean(l0),nanstd(l0)];
            arnn.l1 = l1;
            
            %exitflag
            if nnn==8 && h==0 % plot example session
                ARNN_estimator(W_opt,'wMSE','test',1,0,0);
                [MTp]= RunningMean(y,20);
                x_vec = [1:length(y)];
                figure('Position',[ 0 0 700 200])
                shadedErrorBar(x_vec',MTp(:,1),MTp(:,2),'lineprops','b'); hold on
                plot(x_vec(r==0), y(r==0),'k.'); hold on
                plot(x_vec(r==1), y(r==1),'g.'); hold on
                
                bin = linspace(-.5,.5,20);
                ct0= histc(y(r==0), bin);
                ct1= histc(y(r==1), bin);
                figure; bar(bin,[ct0; ct1],'stacked','EdgeColor','none'); hold on
                xlim([-1 1]); makeaxis();
                pause(1);
                close all
            end
            
            arnn.valid_p=0.2;
            [~,~,~,mse] = LSTM_trainer(y',r',PLOT);   % LSTM GRU model training
            l2 = mse;
            
        catch
            disp('Error! did not create an arnn.')
            arnn=[];
        end
        ARNN{nnn,h+1}= arnn;
        disp(['****** Session:',num2str(nnn),'/',num2str(Nsessions),'******'])
        % [l0,l1,l2]
        Loss{nnn,h+1} = [l0,l1,l2];
        % end
    end
end
%%
AR_coeff=[];
alpha_beta =[];
Nsessions = size(ARNN,1);
for nnn=1:Nsessions %odd: eye, even: hand
    for h=[0,1]
        if isfield(ARNN{nnn,h+1},'exitflag')% && ARNN{nnn,1}.exitflag==1
            AR_coeff = [AR_coeff;[ARNN{nnn,h+1}.AR_coeff ]];
            alpha_beta = [alpha_beta;[ARNN{nnn,h+1}.alpha_beta ]];
            %  L=[L; [ARNN{nnn,h+1}.l0,ARNN{nnn,h+1}.l1]];
        end
    end
end
% plott fitting result
VIOLIN = 1;
figure('Position',[ 0 0 300 600])
subplot(311)
maxNlags = size(AR_coeff,2);
plot([1, maxNlags],zeros(1,2),'k:'); hold on
if VIOLIN
    violin([alpha_beta,AR_coeff],'x',0:1:size(AR_coeff,2),'facecolor',[.3 .3 .3],'edgecolor','none','bw',1e-2,'mc','k','medc','b'); hold on
    legend('off')
else
    plot(AR_coeff','b.'); hold on
end
plot(median(AR_coeff),'b-');
plot(mean(AR_coeff),'k-');
ylim([-0.1 1])
makeaxis();
subplot(312)
L = cell2mat(reshape(Loss,[],1));
%errorbar(1,ARNN{nnn,1}.l0(1),ARNN{nnn,1}.l0(2),'b-','LineWidth',1,'capsize',0); hold on
plot([1 2 3],L,'k-','LineWidth',.5); hold on
%errorbar([1 2], nanmean(L),nanstd(L), 'k-','LineWidth',3,'capsize',5); hold on
[h,pv]= ttest(L(:,1),L(:,2),'Tail','right','Alpha',0.01); % pairwise ttest
text(1, 0.5, ['*** ', num2str(pv,3)] );
[h,pv]= ttest(L(:,3),L(:,2),'Tail','right','Alpha',0.01); % pairwise ttest
text(2, 0.5, ['***', num2str(pv,3)] );
xlim([0.5 3.5])
ylim([.5 1.1])
makeaxis();
subplot(313)
boxplot(L); hold on
makeaxis();
%% Analytical and model variance plot
D = 0.5;
e = linspace(0,1.5,100);
sig0 = 0.1;
sig1 = 0.5;
sig2 = 1;
y_ideal = 2*D*abs(e)./log(abs(D+e)./abs(D-e));
ind_ = (e<D);
y_ideal(ind_) = 0;
ind_ = (e<D);
y_actural = nan(size(e));
y_actural(ind_) = sig1^2*e(ind_).^2 + sig0^2;
y_actural(~ind_) = sig2^2*e(~ind_).^2 + sig0^2;

figure;
plot([-e(end:-1:1),e],[y_ideal(end:-1:1),y_ideal],'k'); hold on
plot([-e(end:-1:1),e],[y_actural(end:-1:1),y_actural],'r'); drawnow
makeaxis('x_label','error(n-1)','y_label','var[e(n)]');