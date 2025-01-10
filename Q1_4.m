
%% Part iv.i): GoogLeNet Feature Extraction & Classification (Experiment 3)


%% Step 1: Clear workspace and load GoogLeNet
clear; clc; close all;

net = googlenet;    %We load the pretrained GoogLeNet



%% Step 2: Define Data Paths and Create imageDatastores
rootFolder = 'Food-11';
trainFolder = fullfile(rootFolder, 'training');
valFolder   = fullfile(rootFolder, 'validation');
testFolder  = fullfile(rootFolder, 'evaluation');


% We choose Approach A: Use all images in each folder

imdsTrain = imageDatastore(trainFolder, ...
    'IncludeSubfolders', true, ...
    'LabelSource','foldernames');

imdsVal   = imageDatastore(valFolder, ...
    'IncludeSubfolders', true, ...
    'LabelSource','foldernames');

imdsTest  = imageDatastore(testFolder, ...
    'IncludeSubfolders', true, ...
    'LabelSource','foldernames');


%% Step 3: We display Basic Info
disp('==== DATASET INFO ====');
disp(['Training set: ', num2str(numel(imdsTrain.Files)), ' images']);
disp(['Validation set: ', num2str(numel(imdsVal.Files)), ' images']);
disp(['Test set: ', num2str(numel(imdsTest.Files)), ' images']);



%% Step 4: Resize Images & Extract Features
inputSize = net.Layers(1).InputSize(1:2);       %GoogLeNet expects 224x224 input images.


augTrain = augmentedImageDatastore(inputSize, imdsTrain);
augVal   = augmentedImageDatastore(inputSize, imdsVal);
augTest  = augmentedImageDatastore(inputSize, imdsTest);


% We picked 'pool5-7x7_s1' for a 1024-dimensional feature vector.

featureLayer = 'pool5-7x7_s1';      %One of the deeper layers

% Extract features
featuresTrain = activations(net, augTrain, featureLayer, 'OutputAs','rows');
featuresVal   = activations(net, augVal,   featureLayer, 'OutputAs','rows');
featuresTest  = activations(net, augTest,  featureLayer, 'OutputAs','rows');

% Get labels
labelsTrain = imdsTrain.Labels;
labelsVal   = imdsVal.Labels;
labelsTest  = imdsTest.Labels;


%% Step 5: Train & Evaluate SVM
disp('Training SVM on GoogLeNet features...');
svmModel = fitcecoc(featuresTrain, labelsTrain);

predTrainSVM = predict(svmModel, featuresTrain);
accTrainSVM  = mean(predTrainSVM == labelsTrain)*100;

predValSVM = predict(svmModel, featuresVal);
accValSVM  = mean(predValSVM == labelsVal)*100;

predTestSVM = predict(svmModel, featuresTest);
accTestSVM  = mean(predTestSVM == labelsTest)*100;

disp(['SVM Accuracy: Train=', num2str(accTrainSVM,4), ...
      '%, Val=', num2str(accValSVM,4), '%, Test=', num2str(accTestSVM,4),'%']);

figure('Name','Confusion Matrix - SVM (GoogLeNet)');
confusionchart(labelsTest, predTestSVM);
title('SVM (GoogLeNet) Confusion Matrix (Test Set)');


%% Step 6: Train & Evaluate KNN
disp('Training KNN on GoogLeNet features...');
kNeighbors = 5;
knnModel = fitcknn(featuresTrain, labelsTrain, 'NumNeighbors', kNeighbors);

predTrainKNN = predict(knnModel, featuresTrain);
accTrainKNN  = mean(predTrainKNN == labelsTrain)*100;

predValKNN = predict(knnModel, featuresVal);
accValKNN  = mean(predValKNN == labelsVal)*100;

predTestKNN = predict(knnModel, featuresTest);
accTestKNN  = mean(predTestKNN == labelsTest)*100;

disp(['KNN(k=', num2str(kNeighbors), '): Train=', num2str(accTrainKNN,4), ...
      '%, Val=', num2str(accValKNN,4), '%, Test=', num2str(accTestKNN,4),'%']);

figure('Name','Confusion Matrix - KNN (GoogLeNet)');
confusionchart(labelsTest, predTestKNN);
title(['KNN (GoogLeNet) Confusion Matrix (Test Set), k=', num2str(kNeighbors)]);


%% Step 7: Train & Evaluate a Shallow Neural Network
disp('Training shallow NN on GoogLeNet features...');
numClasses = numel(categories(labelsTrain));
YTrainIdx  = grp2idx(labelsTrain)';
YTrainOneHot = full(ind2vec(YTrainIdx, numClasses));

hiddenLayerSizes = [100 50];
netNN = patternnet(hiddenLayerSizes);
netNN.trainParam.epochs = 200;
netNN.trainParam.showWindow = true;

[netNN, tr] = train(netNN, featuresTrain', YTrainOneHot);

% Evaluate NN
predTrainNNRaw = netNN(featuresTrain');
[~, predTrainNNIdx] = max(predTrainNNRaw, [], 1);
predTrainNN = categorical(predTrainNNIdx, 1:numClasses, categories(labelsTrain));
accTrainNN  = mean(predTrainNN' == labelsTrain)*100;

predValNNRaw = netNN(featuresVal');
[~, predValNNIdx] = max(predValNNRaw, [], 1);
predValNN = categorical(predValNNIdx, 1:numClasses, categories(labelsVal));
accValNN  = mean(predValNN' == labelsVal)*100;

predTestNNRaw = netNN(featuresTest');
[~, predTestNNIdx] = max(predTestNNRaw, [], 1);
predTestNN = categorical(predTestNNIdx, 1:numClasses, categories(labelsTest));
accTestNN  = mean(predTestNN' == labelsTest)*100;

disp(['NN Accuracy: Train=', num2str(accTrainNN,4), ...
      '%, Val=', num2str(accValNN,4), '%, Test=', num2str(accTestNN,4),'%']);

figure('Name','Confusion Matrix - NN (GoogLeNet)');
confusionchart(labelsTest, predTestNN);
title('Neural Network (GoogLeNet) Confusion Matrix (Test Set)');


%% Step 8: Display Final Summary
disp('==================== FINAL RESULTS (GoogLeNet Feature Extraction) ====================');
disp(['SVM:  Train=', num2str(accTrainSVM,4), '%,  Val=', num2str(accValSVM,4), '%,  Test=', num2str(accTestSVM,4),'%']);
disp(['KNN:  Train=', num2str(accTrainKNN,4), '%,  Val=', num2str(accValKNN,4), '%,  Test=', num2str(accTestKNN,4),'%']);
disp(['NN:   Train=', num2str(accTrainNN,4), '%,  Val=', num2str(accValNN,4), '%,  Test=', num2str(accTestNN,4),'%']);
disp('=======================================================================================');


