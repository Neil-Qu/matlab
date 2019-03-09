%% BSD License

%Copyright (c) 2010, The MathWorks, Inc.
%All rights reserved.
%
%Redistribution and use in source and binary forms, with or without 
%modification, are permitted provided that the following conditions are 
%met:

%    * Redistributions of source code must retain the above copyright 
%      notice, this list of conditions and the following disclaimer.
%    * Redistributions in binary form must reproduce the above copyright 
%      notice, this list of conditions and the following disclaimer in 
%      the documentation and/or other materials provided with the distribution
%    * Neither the name of the The MathWorks, Inc. nor the names 
%      of its contributors may be used to endorse or promote products derived 
%      from this software without specific prior written permission.
      
%THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" 
%AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE 
%IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE 
%ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE 
%LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR 
%CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF 
%SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS 
%INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN 
%CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) 
%ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE 
%POSSIBILITY OF SUCH DAMAGE.

%% Preliminaries

% Clean up our workspace
clear all
clc

% Import our data

White_Wine = dataset('xlsfile', 'White_Wine.xlsx');
X = double(White_Wine(:,1:11));
Y = double(White_Wine(:,12));

% Many of the techniques that we're using make use of 
% random sampling.  Setting the randstream will allow 
% us to reproduce these results precisely.

s = RandStream('mt19937ar','seed',2010);
RandStream.setGlobalStream (s);

%% Perform Exploratory Data Analysis to check assumptions

% See whether the features are normally distribution
% (If the features aren't normally distributed, we shouldn't use
% discriminant analysis)

figure

for i = 1:9
   
   subplot(3,3,i)
   normplot(double(White_Wine(:,i))) 
   title(White_Wine.Properties.VarNames(i))
   
end

%%  Look at Residual Sugar

figure
normplot(double(White_Wine.ResSugar));
title('Residual Sugar');

%% Perform Exploratory Data Analysis to check assumptions

% See whether the features are correlated with one another.
% (If the features are highly correlated we shouldn't use
% Naive Bayes Classifier)

% Covariance Matrix
covmat = corrcoef(double(White_Wine));

figure
x = size(White_Wine, 2);
imagesc(covmat);
set(gca,'XTick',1:x);
set(gca,'YTick',1:x);
set(gca,'XTickLabel',White_Wine.Properties.VarNames);
set(gca,'YTickLabel',White_Wine.Properties.VarNames);
axis([0 x+1 0 x+1]);
grid;
colorbar;

%%  Use a Naive Bayes Classifier to develop a classification model

% Some of the features exhibit significant correlation, however, its
% unclear whether the correlated features will be selected for our model

% Start with a Naive Bayes Classifier

% Use cvpartition to separate the dataset into a test set and a training set
% cvpartition will automatically ensure that feature values are evenly
% divided across the test set and the training set

% Create a cvpartition object that defined the folds
c = cvpartition(Y,'holdout',.2);

% Create a training set

X_Train = X(training(c,1),:);
Y_Train = Y(training(c,1));

%%  Train a Classifier using the Training Set
% missing C:\Program Files\MATLAB\R2009a\toolbox\stats\@NaiveBayes\fit.m

Bayes_Model = fitcnb(X_Train, Y_Train, 'Distribution','kernel');

%%  Evaluate Accuracy Using the Test Set

clc

% Generate a confusion matrix
[Bayes_Predicted] = predict(Bayes_Model, X(test(c,1),:));
[conf, classorder] = confusionmat(Y(test(c,1)),Bayes_Predicted);
conf

% Calculate what percentage of the Confusion Matrix is off the diagonal
Bayes_Error = 1 - trace(conf)/sum(conf(:))


%%  Naive Bayes Classification using Forward Feature Selection

% Create a cvpartition object that defined the folds
c2 = cvpartition(Y,'k',10);

% Set options
opts = statset('display','iter');

fun = @(Xtrain,Ytrain,Xtest,Ytest)...
      sum(Ytest~=predict(fitcnb(Xtrain,Ytrain,'Distribution','kernel'),Xtest));
  
[fs,history] = sequentialfs(fun,X,Y,'cv',c2,'options',opts)
White_Wine.Properties.VarNames(fs)

%% Generate a bagged decision tree

b1 = TreeBagger(250,X,Y,'oobvarimp','on');
oobError(b1, 'mode','ensemble')


%% Rerun the Bagged decision tree with a test set and a training set

b2 = TreeBagger(250,X_Train,Y_Train,'oobvarimp','on');
oobError(b2, 'mode','ensemble')

X_Test = X(test(c,1), :);
Y_Test = Y(test(c,1));

% Use the training classifiers to make Predictions about the test set
[Predicted, Class_Score] = predict(b2,X_Test);
Predicted = str2double(Predicted);
[conf, classorder] = confusionmat(Y_Test,Predicted);
conf

% Calculate what percentage of the Confusion Matrix is off diagonal
Error3 =  1 - trace(conf)/sum(conf(:))


%% Show out of Bag Feature Importance

figure
bar(b1.OOBPermutedVarDeltaError);
xlabel('Feature');
ylabel('Out-of-bag feature importance');
title('Feature importance results');
set(gca, 'XTickLabel',White_Wine.Properties.VarNames(1:11))

%%  Run Treebagger Using Sequential Feature Selection

f = @(X,Y)oobError(TreeBagger(50,X,Y,'method','classification','oobpred','on'),'mode','ensemble');
opt = statset('display','iter');
[fs,history] = sequentialfs(f,X,Y,'options',opt,'cv','none');

%%  Evaluate the accuracy of the model using a performance curve

Test_Results = dataset(Y_Test, Predicted, Class_Score);
[xVal,yVal,~,auc] = perfcurve(Test_Results.Predicted, ...
    Test_Results.Class_Score(:,4),'6'); 

plot(xVal,yVal)
xlabel('False positive rate'); ylabel('True positive rate')
