% Text Sentiment Analysis Yelp
% Scope
% This code aims to:
% 1. Represent descriptions as word2vec matrices
% 2. Train an SVM to correctly classify sentiment based on existing
%    categorical labels
% 3. Train a NN also (task 2)

% Required Matlab add-ons:
% 1. Text Analytics Toolbox
% 2. Text Analytics Toolbox Model for fastText English 16 Billion Token 
%    Word Embedding by MathWorks Text Analytics Toolbox Team
% 3. Statistics and Machine Learning Toolbox

% load pre trained word embeddings

filename = "yelp_labelled.txt"; % Source:
% https://archive.ics.uci.edu/ml/machine-learning-databases/00331/sentiment%20labelled%20sentences.zip

%yelp_emb = trainWordEmbedding(filename)

%words = yelp_emb.Vocabulary;
%V = word2vec(yelp_emb,words);
%XY = tsne(V);
%textscatter(XY,words)

%emb = fastTextWordEmbedding; % class(emb) ~ 'wordEmbedding'

yelpReviews = readtable(filename,'Delimiter','\t','ReadVariableNames',true)

% f = figure;
% f.Position(3) = 1.5*f.Position(3);
% 
% h = histogram(yelpReviews.Sentiment);
% xlabel("Class")
% ylabel("Frequency")
% title("Class Distribution")

cvp = cvpartition(yelpReviews.Sentiment,'Holdout',0.3);
dataTrain = yelpReviews(training(cvp),:);
dataHeldOut = yelpReviews(test(cvp),:);

cvp = cvpartition(dataHeldOut.Sentiment,'HoldOut',0.5);
dataValidation = dataHeldOut(training(cvp),:);
dataTest = dataHeldOut(test(cvp),:);

textDataTrain = dataTrain.Review;
textDataValidation = dataValidation.Review;
textDataTest = dataTest.Review;
YTrain = categorical(dataTrain.Sentiment);
YValidation = categorical(dataValidation.Sentiment);
YTest = categorical(dataTest.Sentiment);

% figure
% wordcloud(textDataTrain);
% title("Training Data")

textDataTrain = lower(textDataTrain);
documentsTrain = tokenizedDocument(textDataTrain);
documentsTrain = erasePunctuation(documentsTrain);

textDataValidation = lower(textDataValidation);
documentsValidation = tokenizedDocument(textDataValidation);
documentsValidation = erasePunctuation(documentsValidation);

documentsTrain(1:5)

enc = wordEncoding(documentsTrain);

documentLengths = doclength(documentsTrain);
figure
histogram(documentLengths)
title("Document Lengths")
xlabel("Length")
ylabel("Number of Documents")

XTrain = doc2sequence(enc,documentsTrain,'Length',30);
XTrain(1:5)

XValidation = doc2sequence(enc,documentsValidation,'Length',30);

inputSize = 1;
embeddingDimension = 100;
numHiddenUnits = enc.NumWords;
hiddenSize = 180;
numClasses = numel(categories(YTrain));

layers = [ ...
    sequenceInputLayer(inputSize)
    wordEmbeddingLayer(embeddingDimension,numHiddenUnits)
    lstmLayer(hiddenSize,'OutputMode','last')
    fullyConnectedLayer(numClasses)
    softmaxLayer
    classificationLayer]

options = trainingOptions('adam', ...
    'MaxEpochs',10, ...    
    'GradientThreshold',1, ...
    'InitialLearnRate',0.01, ...
    'ValidationData',{XValidation,YValidation}, ...
    'Plots','training-progress', ...
    'Verbose',false);

net = trainNetwork(XTrain,YTrain,layers,options);

textDataTest = lower(textDataTest);
documentsTest = tokenizedDocument(textDataTest);
documentsTest = erasePunctuation(documentsTest);

XTest = doc2sequence(enc,documentsTest,'Length',30);
XTest(1:5)

YPred = classify(net,XTest);

accuracy = sum(YPred == YTest)/numel(YPred)