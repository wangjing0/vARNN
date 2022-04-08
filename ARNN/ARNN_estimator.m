function L = ARNN_estimator(W,optMethod,Partial,PLOT,indSession,OPTIM)
% return loss function

global  arnn
pred_error = nan;
if isfield(arnn,'Regularizer')
    Regularization = true;
    sparsity =  arnn.Regularizer.sparsity;
    c1 =  arnn.Regularizer.c1;
    c2 =  arnn.Regularizer.c2;
else % default values
    Regularization =  false;
    sparsity = 1;
    c1 = 3;
    c2 = 3;
end

if isfield(arnn,'valid_p') && isfield(arnn,'Validation') % training and validation
    switch Partial
        case 'train'; Valid = 1.0*(~arnn.Validation);
        case 'test' ; Valid = 1.0* arnn.Validation;
        otherwise;  Valid = ones(size(arnn.Y));
    end
else % default using all data
    Valid = ones(size( arnn.Y));
end

if indSession>=1 && indSession<=size( arnn.Y,2)
    Y =  arnn.Y(:,indSession);
    Reward =  arnn.Reward(:,indSession);
    Valid = Valid(:,indSession);
else % default using all data
    Y =  arnn.Y;
    Reward =  arnn.Reward;
end

AR_coeff = W(1:end-1);
p = length(AR_coeff);
alpha_beta=abs(W(end));
noise= arnn.noise;
alpha = alpha_beta(1); % noise when r=1
beta = 1; % noise when r=0

Nsession = size(Y,2);
y = reshape(Y,[],1);
reward = reshape(Reward,[],1);
valid = reshape(Valid,[],1);

valid(valid==0) = nan; % use nan, instead of zero

if size(Y,2)~=size(Reward,2)
    disp('Error! length does not match.')
    L = nan;
    return;
else
    %     for n = 1:Nsession
    %         y = Y(:,n);
    %         reward = Reward(:,n);
    y_target = y(p+1:end);
    reward_target = reward(p+1:end);
    valid = valid(p+1:end);
    Nsample = length(y_target);
    y_regressor = nan(Nsample,p); % embedded
    reward_regressor = nan(Nsample,p);
    y_predictor = nan(size(y_target));
    for ip=1:p
        y_regressor(:,ip) = y(ip:ip+Nsample-1) ;
        reward_regressor(:,ip) = reward(ip:ip+Nsample-1) ;
    end
    variance_regressor = (beta.*(1-reward_regressor) + alpha.*(reward_regressor)); % variance contribution, reward dependent.
    y_predictor =  y_regressor * AR_coeff(end:-1:1)'; % estimated mean
    se = ((y_predictor - y_target).^2); % squared error
    
    switch optMethod
        case 'MSE' % mean squared error
            pred_error = nanmean(se.*valid);% MSE, pred_error
        case 'rwMSE'% randomly weighted, sample from a uniform distribution, return a set of Loss values
            Nw_sample = 1e2;
            weight = rand(Nsample,Nw_sample);% number of data points x number of weight sets  ,
            pred_error = nanmean((repmat(se,1,Nw_sample).*valid.* weight)./nanmean(valid.*weight));
        case 'wMSE'  % reward modulated, weighted MSE,
            var_weight =  (1./variance_regressor) * (AR_coeff(end:-1:1).^2)' ;% weights are modulated by reward
            pred_error =  nanmean(valid.*var_weight.*se./nanmean(valid.*var_weight)); % using inverse-variance weighting
        otherwise
            disp('Error! specify optimization method');
            pred_error = nan;
    end
    
    if  Regularization && strcmpi(Partial,'train')
        if 0
            panelty_coeff = nanmean(sqrt(abs(AR_coeff))); % for unnormalized data
        else
            panelty_coeff = (nanmean(2./(1+exp(-c1.*(abs(AR_coeff).^(1/c2)))))-1); % for nomalized data
        end
        panelty_noise = (alpha/beta + beta/alpha);
        lambda_coeff = 1e-2;
        panelty = lambda_coeff.*((1./(sparsity - 1e-3) -1)* panelty_coeff + panelty_noise)*(pred_error);
    else
        panelty = 0;
    end
     % L =  R^2 , fraction of unexplained variance/total variance
    total_var = nanvar(y_target);
    if OPTIM % during optimization, with panelty for regularization
        L = (pred_error + panelty)./total_var;
    else
        L = (pred_error)./total_var;
    end
    if PLOT
        figure('Position',[ 0 0 700 200])
        subplot(1,4,1); bar(arnn.AR_coeff,'EdgeColor','none','FaceAlpha',.5); hold on; plot(1:p,AR_coeff,'k.'); hold on; makeaxis();
        subplot(1,4,[2:4]);
        x_vec = 1:min(Nsample,1e3);
        r_vec = reward_target(x_vec)>0;
        shadedErrorBar(x_vec',y_predictor(x_vec),sqrt(total_var*(1-L)./var_weight(x_vec)),'lineprops','b'); hold on
        % plot(sqrt(total_var*(1-L)./var_weight(x_vec)),'k-'); hold on
        % stem(reward(x_vec+p)); hold on
        plot(x_vec(r_vec), y_target(x_vec(r_vec)),'g.'); hold on
        plot(x_vec(~r_vec), y_target(x_vec(~r_vec)),'k.'); hold on
        legend({'one step forcasting','simulation'});
    end
    
    return;
end