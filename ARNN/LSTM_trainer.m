function [LSTMnet,Xinit,Y_pred,mse] = LSTM_trainer(Y,Reward,PLOT)
global arnn
if ~isfield(arnn, 'tau')
    arnn.tau = 1;
    arnn.d   = 20;
end
if ~isfield(arnn, 'valid_p')
    arnn.valid_p = 0.2; % percent for validation
end

XTrain = [];
YTrain = [];
XTest = [];
YTest = [];
indexSession = [];
Xinit = [];
split_ratio = 1 - arnn.valid_p; % perctile for training
Nsession = size(Y,2);
for sss=1:Nsession
    y = reshape(Y(:,sss),[],1);
    r = reshape(Reward(:,sss),[],1);
    [Xtrain,Ytrain,Xtest,Ytest,indexTrain,indexTest] = getDataSetLSTM(y,r,arnn.tau,arnn.d,'prediction_steps',1,'split_ratio',split_ratio);
    XTrain = [XTrain;Xtrain];
    YTrain = [YTrain;Ytrain];
    XTest = [XTest;Xtest];
    YTest = [YTest;Ytest];
    indexSession = [indexSession;sss*ones(length(Ytest),1)];
    Xinit = [Xinit;Xtrain(1)];
end

% create nn layers
numFeatures = 2; % inputsize, x and reward, a sequence or one
numHiddenUnits = 32; %
numResponses = 1; % one output

layers = [sequenceInputLayer(numFeatures,'Name','Input') % input
    gruLayer(numHiddenUnits,'OutputMode','last','InputWeightsInitializer','he','Name','GRU')  % GRU 
    %lstmLayer(numHiddenUnits,'OutputMode','last','InputWeightsInitializer','he','Name','LSTM') % LSTM
    fullyConnectedLayer(numResponses,'Name','FullyConnect')
    regressionLayer('Name','RegOutput')];

lgraph = layerGraph(layers);
%figure; plot(lgraph)

maxEpochs = 1e3;
miniBatchSize = min(256,length(XTrain));

options = trainingOptions('adam', ...
    'InitialLearnRate',0.003,...
    'L2Regularization',0.001,...
    'ExecutionEnvironment','cpu', ...
    'MaxEpochs',maxEpochs, ...
    'Shuffle','every-epoch',...
    'ValidationData',{XTest,cell2mat(YTest)},...
    'ValidationFrequency',10,...
    'ValidationPatience',20,...
    'MiniBatchSize',miniBatchSize, ...
    'GradientThreshold',1, ...
    'Verbose',false);

if PLOT
    options.Plots = 'training-progress';
else
    options.Plots = 'none';
end
% train the net
LSTMnet = trainNetwork(XTrain,cell2mat(YTrain),layers,options);

% output prediction after training
Y_pred = [];
mse = [];
if Nsession==1
    [Xtrain,Ytrain] = getDataSetLSTM(y,r,arnn.tau,arnn.d,'prediction_steps',1,'split_ratio',1);
    Y_pred = predict(LSTMnet,Xtrain);
    Y_ = cell2mat(Ytrain);
    mse =  (nanmean((Y_ - Y_pred).^2))/nanvar(Y_);
end

end