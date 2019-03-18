% Load a pretrained word embedding.

% load pre trained word embeddings
emb = fastTextWordEmbedding; % class(emb) ~ 'wordEmbedding'

% Load an opinion lexicon listing positive and negative words.
% Read positive words NB lexicon files must be at script directory level
% fopen returns a file id
fidPositive = fopen('positive-words.txt');
% textscan returns a cell array...
C = textscan(fidPositive,'%s','CommentStyle',';');
% ...which we can examine with celldisp(C) or by indexing e.g. C{1}{2}
% convert cell array to string
wordsPositive = string(C{1}); % class(wordsPositive) ~ 'string'
% Read negative words
%fidNegative = fopen(fullfile('opinion-lexicon-English','negative-words.txt'));
fidNegative = fopen('negative-words.txt');
C = textscan(fidNegative,'%s','CommentStyle',';');
wordsNegative = string(C{1}); % class(wordsNegative) ~ 'string'
% Close all open files
fclose all;
% Concatenate string with labeled words
words = [wordsPositive;wordsNegative]; % class(words) ~ 'string'
% create labels, initialise table cells to NaN
labels = categorical(nan(numel(words),1)); % class(labels) ~ 'categorical'
% from start to number of positive words, assign "Positive" category to cells
labels(1:numel(wordsPositive)) = "Positive";
% from end of number of positive words plus one, to end of labels' table, 
% assign "Negative" category to cells
labels(numel(wordsPositive)+1:end) = "Negative";
% create a data table with columns and column headers
data = table(words,labels,'VariableNames',{'Word','Label'}); % class(data) ~ 'table'

%View the first few words labeled as positive.
idx = data.Label == "Positive"; % class(idx) ~ 'logical'
head(data(idx,:))
%View the first few words labeled as negative.
idx = data.Label == "Negative";
head(data(idx,:))

% Prepare Data for Training
% To train the sentiment classifier, convert the words to word vectors using the pretrained 
% word embedding emb. First remove the words that do not appear in the word embedding emb.

idx = ~isVocabularyWord(emb,data.Word);
data(idx,:) = [];
%Set aside 10% of the words at random for testing.
numWords = size(data,1);
%  define a random partition on a set of data of a specified size
cvp = cvpartition(numWords,'HoldOut',0.1);
dataTrain = data(training(cvp),:); % class(training(cvp)) ~ 'logical'
dataTest = data(test(cvp),:); % class(dataTest) ~ 'table'
% Set training words aside
wordsTrain = dataTrain.Word; % class(wordsTrain) ~ 'string' (vector)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Convert words to vector representations%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% See "Efficient Estimation of Word Representations inVector Space"
% https://arxiv.org/pdf/1301.3781.pdf

%Convert the words in the training data to word vectors using word2vec.
XTrain = word2vec(emb,wordsTrain); % class(XTrain) ~ 'single'
% NB word2vec returns Matrix of word embedding vectors
YTrain = dataTrain.Label; % class(YTrain) ~ 'categorical'

% train classifier 
% NB Documentation mentions different kernels may be used, as well as
% hyperparameters - TODO examine options as examiner asks for
% hyperparameter tuning

mdl = fitcsvm(XTrain,YTrain); % class(mdl) ~ 'ClassificationSVM'

% mdl = 

%  ClassificationSVM
%             ResponseName: 'Y'
%    CategoricalPredictors: []
%               ClassNames: [Positive    Negative]
%           ScoreTransform: 'none'
%          NumObservations: 5872
%                    Alpha: [754×1 single]
%                     Bias: -0.0032
%         KernelParameters: [1×1 struct]
%           BoxConstraints: [5872×1 double]
%          ConvergenceInfo: [1×1 struct]
%          IsSupportVector: [5872×1 logical]
%                   Solver: 'SMO'
                                     
% test classifier
wordsTest = dataTest.Word; % class(wordsTest) ~ 'string' (vector)
XTest = word2vec(emb,wordsTest); % class(XTest) ~ 'single' (vector) 
YTest = dataTest.Label; % class(YTest) ~ 'categorical'

%Predict the sentiment labels of the test word vectors.
[YPred,scores] = predict(mdl,XTest); % class(YPred) ~ 'categorical'
                                     % class(scores) ~ 'single' (vector)
%Visualize the classification accuracy in a confusion matrix.
figure
confusionchart(YTest,YPred);

%%%%%%%%%%%%%%%%%%%%
% Word Cloud Debug %
%%%%%%%%%%%%%%%%%%%%

%Visualize the classifications in word clouds. Plot the words with positive and negative sentiments in word clouds with word sizes corresponding to the prediction scores.
%figure
%subplot(1,2,1)
%idx = YPred == "Positive";
%wordcloud(wordsTest(idx),scores(idx,1));
%title("Predicted Positive Sentiment")

%subplot(1,2,2)
%wordcloud(wordsTest(~idx),scores(~idx,2));
%title("Predicted Negative Sentiment")

% Calculate Sentiment of Collections of Yelp Reviews 
% NB This has performed poorly with the SVM classifier. One reason may be 
% the positive, negative words' lexicon does not have good coverage in the
% Yelp corpus. The UCI "Sentiment Labelled Sentences Data Set" was created
% for "From Group to Individual Labels using Deep Features" 
% http://mdenil.com/media/papers/2015-deep-multi-instance-learning.pdf
% which AFAIK does cite source for list of positive and negative terms

% filename = "sentiment-labelled-sentences/yelp_labelled.txt";
% dataReviews = readtable(filename,'TextType','string');
% textData = dataReviews.Review;
% textData(1:10)
% Load a pretrained word embedding
% emb = fastTextWordEmbedding;
% documents = preprocessReviews(textData);

%idx = ~isVocabularyWord(emb,documents.Vocabulary);

%documents = removeWords(documents,idx);

%words = documents.Vocabulary;
%words(ismember(words,wordsTrain)) = [];

%vec = word2vec(emb,words);
%figure
%subplot(1,2,1)
%idx = YPred == "Positive";
%wordcloud(words(idx),scores(idx,1));
%title("YELP Predicted Positive Sentiment")
%subplot(1,2,2)
%wordcloud(words(~idx),scores(~idx,2));
%title("YELP Predicted Negative Sentiment")

%  __   __  ____  ____  __ _  ____ 
% / _\ (  )(  _ \(  _ \(  ( \(  _ \
%/    \ )(  )   / ) _ (/    / ) _ (
%\_/\_/(__)(__\_)(____/\_)__)(____/
%

% http://insideairbnb.com/get-the-data.html
filename = "reviews.csv"; % Source:
% http://data.insideairbnb.com/united-states/ma/boston/2019-02-09/data/reviews.csv.gz

dataReviews = readtable(filename,'TextType','string'); % class(dataReviews) ~ 'table'
textData = dataReviews.comments; % class(textData) ~ 'string'
textData(1:10)

documents = preprocessReviews(textData); % class(documents) ~ 'tokenizedDocument' 

% From https://uk.mathworks.com/help/textanalytics/ref/tokenizeddocument.html
% Vocabulary (Property of tokenizedDocument) — Unique words in the documents
% Unique words in the documents, specified as a string array. 
% The words do not appear in any particular order.

% As per sentiment words, create logical vector for existing and
% non-existing words
idx = ~isVocabularyWord(emb,documents.Vocabulary);
% remove words by specifying the numeric or logical indices (idx)
documents = removeWords(documents,idx); 

words = documents.Vocabulary; % class(words) ~ 'string' (vector)
% ismember(A,B) returns an array containing logical 1 (true) where the data 
% in A is found in B. Elsewhere, the array contains logical 0 (false)
% NB "= []" assignment empties entries where logical array index = 0
words(ismember(words,wordsTrain)) = []; % "clean" words string vector

vec = word2vec(emb,words); % class(vec) ~ 'single' vector
[YPred,scores] = predict(mdl,vec); % class(YPred) ~ 'categorical'
                                   % class(scores) ~ 'single' (vector)

figure
subplot(1,2,1)
idx = YPred == "Positive";
wordcloud(words(idx),scores(idx,1));
title("Predicted Positive Sentiment")
subplot(1,2,2)
wordcloud(words(~idx),scores(~idx,2));
title("Predicted Negative Sentiment")

idx = [7 34 331 1788 1820 1831 2185 21892 63734 76832 113276 120210];
for i = 1:numel(idx)
 words = string(documents(idx(i)));
 vec = word2vec(emb,words);
 [~,scores] = predict(mdl,vec);
 sentimentScore(i) = mean(scores(:,1));
end

[sentimentScore' textData(idx)]
