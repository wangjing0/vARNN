function [XTrain,YTrain,XTest,YTest,indexTrain,indexTest] = getDataSetLSTM(y,r,tau,d,varargin)
steps  = 1; % prdiction forward steps
split_ratio = 0.8; % train and validation

iVarArg = 1;
while iVarArg <= length(varargin)
    argOkay = true;
    switch varargin{iVarArg}
        case 'split_ratio'  % split to training and test
            split_ratio = varargin{iVarArg+1}; iVarArg = iVarArg + 1;
        case 'prediction_steps'
            steps = varargin{iVarArg+1}; iVarArg = iVarArg + 1;            
        otherwise
            argOkay = false;
    end
    if ~argOkay
        disp(['Ignoring invalid argument #' num2str(iVarArg+1)]);
    end
    iVarArg = iVarArg + 1;
end

order = 1:length(y);
%for ns = 1:size(Y,2)
    %clear y r
   % y = arnn.Y(:,ns);
   % r = arnn.Reward(:,ns);
    nnan_indx = find(~isnan(r) & ~isnan(y));
    y = y(nnan_indx);
    r = r(nnan_indx);
    order = order(nnan_indx);
   % y = zscore(y); % normalize data within session
    
    [X,Y] = DelayEmbeddingZ(y,tau,d,'prediction_steps', steps);
    [R] = DelayEmbeddingZ(r,tau,d,'prediction_steps', steps);
    
    y_output = Y(:,1:steps);
    X = permute(cat(3,X,R),[1 3 2]);%  number of sequences x numInputs(features) x SeqLength
    %clear y r X Y R
    nSample = size(X,1);
    XX = cell(nSample,1);
    YY = cell(nSample,1);
    
    for sss=1:nSample
        XX{sss} = squeeze(X(sss,:,:));
        YY{sss} = [y_output(sss,:)];        
    end
    %YTrain = y_output;
   % XX = [XX; X_];
   % YY = [YY; Y_];   

%end
nSeq = length(XX); % number of sequences


train_label = rand(nSeq,1)<=split_ratio;
XTrain = XX(train_label);
YTrain = YY(train_label);
indexTrain = order(train_label);
XTest = XX(~train_label);
YTest = YY(~train_label);
indexTest = order(~train_label);
end


