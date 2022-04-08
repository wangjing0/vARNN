function arnn = setup_arnn(y,r,maxNlags)
if isvector(y) && length(y)==length(r) && length(y)>maxNlags
   if isrow(y) 
       y = y'; r = r';
   end 
Nsample = length(y);
Ntrial = 1;
Nsession = 1;
noise = nanstd(y);

[pc,~] = parcorr(y,maxNlags); % partial correlation, as the initial values for ar coefficients
arnn=struct('Nsample',Nsample,'Ntrial',Ntrial,'Nsession',Nsession,...
                  'AR_coeff',[],'parcor',pc(2:end)',...
                  'alpha_beta',[], 'noise',noise,'k_reward',nan);
              
arnn.Regularizer.sparsity = .9;
arnn.Regularizer.c1 = 3;
arnn.Regularizer.c2 = 3;
arnn.Y = y;
arnn.Reward = r;              
else
    disp('Error! Fail to setup arnn')
    arnn=[];
end
return;