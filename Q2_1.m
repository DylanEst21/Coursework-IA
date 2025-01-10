
%% Section 1: Load and Preprocess Data

clear; close all; clc;

%We define the data paths we are going to need
dataFolder = "smartphone_data"; 
trainFolder = fullfile(dataFolder, "Train");
testFolder = fullfile(dataFolder, "Test");

% Load activity labels from file
activityLabelsPath = fullfile(dataFolder, "activity_labels.txt");
activityLabels = readtable(activityLabelsPath, 'ReadVariableNames', false);
activityLabels.Properties.VariableNames = {'ActivityID', 'ActivityName'};

%Feature Names Extraction
featuresPath = fullfile(dataFolder, "features.txt");
featureLines = strtrim(readlines(featuresPath));
featureNames = featureLines(~strcmp(featureLines, ""));

fprintf('Number of variables in features.txt: %d\n', length(featureNames));

%We load training and testing data
X_train = readmatrix(fullfile(trainFolder, "X_train.txt"));
y_train = readmatrix(fullfile(trainFolder, "y_train.txt"));
X_test = readmatrix(fullfile(testFolder, "X_test.txt"));
y_test = readmatrix(fullfile(testFolder, "y_test.txt"));


fprintf('Training Data Size: %d samples, %d features\n', size(X_train, 1), size(X_train, 2));
fprintf('Testing Data Size: %d samples, %d features\n', size(X_test, 1), size(X_test, 2));


%We convert labels to categorical
Y_train = categorical(y_train, activityLabels.ActivityID, activityLabels.ActivityName);
Y_test = categorical(y_test, activityLabels.ActivityID, activityLabels.ActivityName);

%The features are already normalized according to the information in README.txt

%Class Balancing by Oversampling Minority Classes
fprintf("Balancing classes by oversampling minority classes...\n");
counts = countcats(Y_train);
maxCount = max(counts);
categoriesList = categories(Y_train);
X_balanced = [];
Y_balanced = [];

for i = 1:length(categoriesList)
    currentCategory = categoriesList{i};
    idx = find(Y_train == currentCategory);
    currentCount = length(idx);
    if currentCount < maxCount

        %Oversample
        numToSample = maxCount - currentCount;
        sampledIdx = randsample(idx, numToSample, true);
        X_balanced = [X_balanced; X_train(sampledIdx, :)]; 
        Y_balanced = [Y_balanced; Y_train(sampledIdx)];
    end

    %Add original samples
    X_balanced = [X_balanced; X_train(idx, :)]; 
    Y_balanced = [Y_balanced; Y_train(idx)];
end

% Verify class distribution after balancing
disp("Post-balancing class distribution:");
disp(countcats(Y_balanced));

%We Split test data into validation and final test sets (50%-50%) using stratified split
fprintf("Splitting test data into validation and final test sets...\n");
categoriesList = categories(Y_test);
X_val = [];
Y_val = [];
X_test_final = [];
Y_test_final = [];

for i = 1:length(categoriesList)
    currentCategory = categoriesList{i};
    idx = find(Y_test == currentCategory);
    numSamples = length(idx);
    numVal = round(0.5 * numSamples);

    shuffledIdx = idx(randperm(numSamples));
    valIdx = shuffledIdx(1:numVal);
    testIdx = shuffledIdx(numVal+1:end);

    X_val = [X_val; X_test(testIdx, :)];         
    Y_val = [Y_val; Y_test(testIdx)];

    X_test_final = [X_test_final; X_test(valIdx, :)];  
    Y_test_final = [Y_test_final; Y_test(valIdx)];
end

disp("Validation Set Class Distribution:");
disp(countcats(Y_val));
disp("Final Test Set Class Distribution:");
disp(countcats(Y_test_final));

%Create sliding window sequences
sequenceLength = 5;     %Num of consecutive samples per sequence
stride = 1;     %Step size for sliding window

fprintf("Creating sliding window sequences...\n");
numSequences_train = floor((size(X_balanced, 1) - sequenceLength) / stride) + 1;
X_train_seq = cell(1, numSequences_train);
Y_train_seq = categorical(strings(numSequences_train, 1));

for i = 1:numSequences_train
    startIdx = (i-1)*stride + 1;
    endIdx = startIdx + sequenceLength - 1;
    currentSeq = X_balanced(startIdx:endIdx, :);
    X_train_seq{i} = currentSeq';       %transpose to [features x time steps]
    Y_train_seq(i) = Y_balanced(endIdx);        %assign label of the last sample in the sequence
end

numSequences_val = floor((size(X_val, 1) - sequenceLength) / stride) + 1;
X_val_seq = cell(1, numSequences_val);
Y_val_seq = categorical(strings(numSequences_val, 1));

for i = 1:numSequences_val
    startIdx = (i-1)*stride + 1;
    endIdx = startIdx + sequenceLength - 1;
    currentSeq = X_val(startIdx:endIdx, :);
    X_val_seq{i} = currentSeq'; 
    Y_val_seq(i) = Y_val(endIdx);   
end

numSequences_test = floor((size(X_test_final, 1) - sequenceLength) / stride) + 1;
X_test_seq = cell(1, numSequences_test);
Y_test_seq = categorical(strings(numSequences_test, 1));

for i = 1:numSequences_test
    startIdx = (i-1)*stride + 1;
    endIdx = startIdx + sequenceLength - 1;
    currentSeq = X_test_final(startIdx:endIdx, :);
    X_test_seq{i} = currentSeq'; 
    Y_test_seq(i) = Y_test_final(endIdx);   
end



% Display sequence dimensions
fprintf('Training Sequences: %d sequences, each of size [%d x %d]\n', ...
    length(X_train_seq), size(X_train_seq{1},1), size(X_train_seq{1},2));
fprintf('Validation Sequences: %d sequences, each of size [%d x %d]\n', ...
    length(X_val_seq), size(X_val_seq{1},1), size(X_val_seq{1},2));
fprintf('Final Test Sequences: %d sequences, each of size [%d x %d]\n', ...
    length(X_test_seq), size(X_test_seq{1},1), size(X_test_seq{1},2));


% Display a sample training sequence
figure;
imagesc(X_train_seq{1});
title('Sample Training Sequence');
xlabel('Time Steps');
ylabel('Features');
colorbar;


%% Section 2: Define LSTM Network Architecture

%We define number of features and classes
numFeatures = size(X_train_seq{1}, 1);      %Num of features (561)
numClasses = numel(categories(Y_train));        %Num of activity classes


%We define the LSTM network architecture
layers = [
    sequenceInputLayer(numFeatures, 'Name', 'Input')
    lstmLayer(100, 'OutputMode', 'last', 'Name', 'LSTM')        %100 hidden units
    dropoutLayer(0.2, 'Name', 'Dropout')        %20% dropout rate
    fullyConnectedLayer(numClasses, 'Name', 'FC') 
    softmaxLayer
    classificationLayer]; 



%% Section 3: Specify Training Options

%We define the training options
maxEpochs = 10; 
miniBatchSize = 64; 

options = trainingOptions('adam', ...
    'MaxEpochs', maxEpochs, ...
    'MiniBatchSize', miniBatchSize, ...
    'InitialLearnRate', 0.001, ...
    'GradientThreshold', 1, ...
    'Shuffle', 'every-epoch', ...
    'ValidationData', {X_val_seq, Y_val_seq}, ...
    'ValidationFrequency', 30, ...
    'Verbose', true, ...       
    'VerboseFrequency', 30, ... 
    'Plots', 'training-progress');   


%% Section 4: Train the LSTM Network

fprintf("Training the LSTM network...\n");
net = trainNetwork(X_train_seq, Y_train_seq, layers, options);


%% Section 5: Evaluate the Network on Test Data

fprintf("Evaluating the network on test data...\n");
y_pred = classify(net, X_test_seq, 'MiniBatchSize', miniBatchSize);

% Calculate accuracy
accuracy = sum(y_pred == Y_test_seq) / numel(Y_test_seq);
fprintf('Test Accuracy: %.2f%%\n', accuracy * 100);

% Generate confusion matrix
confMat = confusionmat(Y_test_seq, y_pred, 'Order', activityLabels.ActivityName);

% Display confusion matrix
figure;
confusionchart(confMat, categories(Y_test_seq));
title('Confusion Matrix on Test Data');


%% Section 6: Analyze Results

%We calculate Precision, Recall, and F1-Score for each class
numClasses = size(confMat, 1);
precision = zeros(numClasses,1);
recall = zeros(numClasses,1);
f1Score = zeros(numClasses,1);

for i = 1:numClasses
    TP = confMat(i,i);
    FP = sum(confMat(:,i)) - TP;
    FN = sum(confMat(i,:)) - TP;

    if (TP + FP) == 0
        precision(i) = 0;
    else
        precision(i) = TP / (TP + FP);
    end

    if (TP + FN) == 0
        recall(i) = 0;
    else
        recall(i) = TP / (TP + FN);
    end

    if (precision(i) + recall(i)) == 0
        f1Score(i) = 0;
    else
        f1Score(i) = 2 * (precision(i) * recall(i)) / (precision(i) + recall(i));
    end
end


%Handle NaN cases where division by zero might occur
precision(isnan(precision)) = 0;
recall(isnan(recall)) = 0;
f1Score(isnan(f1Score)) = 0;


%We calculate overall accuracy
accuracy = sum(diag(confMat)) / sum(confMat(:));


%Create metrics table using activityLabels.ActivityName
metricsTable = table(activityLabels.ActivityName, precision, recall, f1Score, ...
    'VariableNames', {'Class', 'Precision', 'Recall', 'F1_Score'});


overallRow = table("Overall", accuracy, accuracy, accuracy, ...
    'VariableNames', {'Class', 'Precision', 'Recall', 'F1_Score'});
metricsTable = [metricsTable; overallRow];


disp("Classification Metrics:");
disp(metricsTable);


%We want to identify the most confused classes
confMatOffDiag = confMat;
confMatOffDiag(1:size(confMat,1)+1:end) = 0;    %zero out the diagonal
[maxConfusion, idxMaxConfusion] = max(confMatOffDiag(:));


if maxConfusion > 0
    [row, col] = ind2sub(size(confMatOffDiag), idxMaxConfusion);
    fprintf('Most confused classes: %s and %s with %d misclassifications.\n', ...
        string(activityLabels.ActivityName(row)), string(activityLabels.ActivityName(col)), maxConfusion);
else
    fprintf('No significant misclassifications found.\n');
end



