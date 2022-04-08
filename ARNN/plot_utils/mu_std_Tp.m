function [mTp_stdTp,Tp,b_mu,b_sigma] = mu_std_Tp(tp,tp_1,kreward,reward,ts,Nend,REWARD,Nsample)
% tp_1: tp(n-1)
% tp: tp(n)
% reward: reward(n-1)
%Nsample = 200;
if length(tp)==length(tp_1) && length(tp)==length(reward)
    if iscolumn(tp); tp = tp';end
    if iscolumn(tp_1); tp_1 = tp_1';end
    if iscolumn(reward); reward = reward';end
if REWARD
            mTp_stdTp(:,1)=[ nanmean(tp_1(reward==0 & tp_1 <0));  nanmean(tp_1(reward>0 & tp_1 <0)); nanmean(tp_1(reward>0 & tp_1>0)); nanmean(tp_1(reward==0 & tp_1 >0))];
            mTp_stdTp(:,2)=[ nanmean(tp(reward==0 & tp_1 <0));  nanmean(tp(reward>0 & tp_1 <0)); nanmean(tp(reward>0 & tp_1>0)); nanmean(tp(reward==0 & tp_1 >0))];
            mTp_stdTp(:,3)=[ nanstd(tp(reward==0 & tp_1 <0),1);  nanstd(tp(reward>0 & tp_1 <0),1); nanstd(tp(reward>0 & tp_1>0)); nanstd(tp(reward==0 & tp_1 >0),1)];
   Tp={};
else
[tp_1,sortInd] = sort(tp_1);
tp = tp(sortInd);
reward = reward(sortInd);
mTp_stdTp=[];Tp={};
Nboots = 100;
if sum(abs(kreward))~=0
    for k=Nend:length(kreward)-Nend
        indx = find((tp_1 - ts) >= kreward(k) & (tp_1 - ts) < kreward(k+1) & ~isnan(tp_1));
        if Nsample % number of subsamples
            ind = indx(randperm(length(indx),min(length(indx),Nsample)));
        else
            ind = indx;            
        end
        tp__ = sort(tp(ind),'ascend');
        [mu,sig]=subsample(tp__,Nboots);
    %    mTp_stdTp=[ mTp_stdTp;[nanmean(tp_1(ind)),nanmean(tp__),nanstd(tp__,1),nanmean(reward(ind)), nanstd(mu),nanstd(sig)] ];
   mTp_stdTp=[ mTp_stdTp;[nanmean(tp_1(ind)),nanmean(tp__),nanstd(tp__,1),nanmean(reward(ind)),...
       nanmedian(mu)- prctile(mu,1), prctile(mu,99)-nanmedian(mu),nanmedian(sig)-prctile(sig,1), prctile(sig,99)-nanmedian(sig),sqrt(nanmean(tp__.^2))] ];
   Tp = [Tp;{[tp_1(ind);tp(ind);reward(ind)]}];
    end
else
    Nbin=length(kreward); % number of bins
    indexEdge =ceil(linspace(1, length(tp),Nbin+1));indexEdge(end) = length(tp);
    for k=Nend:(Nbin-Nend+1)
        clear tp__
        indx = indexEdge(k):indexEdge(k+1);
        
        if Nsample
            ind = indx(randperm(length(indx),min(length(indx),Nsample)));
        else
            ind = indx;
            
        end
        
        tp__ = sort(tp(ind),'ascend');
        [mu,sig]=subsample(tp__,Nboots);
      %  mTp_stdTp=[ mTp_stdTp;[nanmean(tp_1(ind)),nanmean(tp__),nanstd(tp__,1),nanmean(reward(ind)), nanstd(mu),nanstd(sig)] ];
   mTp_stdTp=[ mTp_stdTp;[nanmedian(tp_1(ind)),nanmean(tp__),nanstd(tp__,1),nanmean(reward(ind)),...
       nanmedian(mu)- prctile(mu,1), prctile(mu,99)-nanmedian(mu),nanmedian(sig)-prctile(sig,1), prctile(sig,99)-nanmedian(sig),sqrt(nanmean(tp__.^2))] ];
   Tp = [Tp;{[tp_1(ind);tp(ind);reward(ind)]}];
    end
    
end

% linear and quadratic model 
try
x = mTp_stdTp(:,1); mu = mTp_stdTp(:,2); sigma = mTp_stdTp(:,3);
indx_nan = sum(isnan(mTp_stdTp),2);
[b_mu.b,b_mu.bint,b_mu.r,b_mu.rint,b_mu.stats] = regress(mu(~indx_nan),[x(~indx_nan) ones(size(x(~indx_nan)))],0.05);
[b_sigma.b,b_sigma.bint,b_sigma.r,b_sigma.rint,b_sigma.stats]= regress(sigma,[x.^2 x ones(size(x))],0.05);
end
end

else
    disp('Error, tp  tp-1 dimension does not match')
end


end

function  [mu,sig]=subsample(t,Nboots)
N =length(t);
mu = nan(N,1);
sig = nan(N,1);
%rand('state',1);
t = t(randperm(N,N));
%Nboots = ceil(N./100);
%Tedges = ceil(linspace(1,N,Nboots+1)); Tedges(end) = N;
for i=1:Nboots
    inddd = randperm(N,ceil(N.*0.5));%%randperm(N,ceil(N.*0.5));% Tedges(i):Tedges(i+1);%%randperm(N,ceil(N.*0.5));%
    mu(i) = nanmean(t(inddd));
    sig(i) = nanstd(t(inddd));
end
end