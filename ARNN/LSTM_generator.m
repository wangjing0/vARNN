function [Yout,Reward] = LSTM_generator(LSTMnet,Xinit,SeqLen,k_reward,dk_reward,n0)
nSeqInput = size(Xinit{1,1},2); % input sequence length
nTrial = length(Xinit);
Yout = nan(nTrial,SeqLen);% SeqLen: forward prediction length
Reward = nan(nTrial,SeqLen);
X = Xinit;
kreward = repmat(k_reward,nTrial,1);
for sss=1:SeqLen
    Yout(:,sss) = predict(LSTMnet,X) + n0*randn(nTrial,1);
    Reward(:,sss) = abs(Yout(:,sss)) < kreward;
    kreward = kreward + dk_reward.*(Reward(:,sss)==0) - dk_reward.*(Reward(:,sss)==1); % on staircase
    kreward(kreward<=k_reward) = k_reward;
    X_temp = cell2mat(X);
    X_new = [X_temp,[Yout(:,sss);Reward(:,sss)]];
    X = mat2cell(X_new(:,end-nSeqInput+1:end), 2*ones(1,nTrial),nSeqInput);%{X_new(:,end-nSeqInput+1:end)};
end
Yout = [Yout' - nanmean( Yout')]';

end
