
%% Part iv.ii): Transfer Learning with GoogLeNet (Experiment 4)
% We have a layer graph "lgraph_1" exported from the tool
% Deep Network Designer, containing the final layers for 11 classes.

clear; clc; close all;

%% Step 1: Load the modified network structure
if exist('modified_googlenet_layers.mat','file')
    load('modified_googlenet_layers.mat','lgraph_1');
else
    error('Modified GoogLeNet file not found. ');
end




%% Step 2: Create Training, Validation, and Test Datastores
rootFolder = 'Food-11';  

trainFolder = fullfile(rootFolder, 'training');
valFolder   = fullfile(rootFolder, 'validation');
testFolder  = fullfile(rootFolder, 'evaluation');

imdsTrain = imageDatastore(trainFolder, ...
    'IncludeSubfolders', true, ...
    'LabelSource','foldernames');

imdsVal   = imageDatastore(valFolder, ...
    'IncludeSubfolders', true, ...
    'LabelSource','foldernames');

imdsTest  = imageDatastore(testFolder, ...
    'IncludeSubfolders', true, ...
    'LabelSource','foldernames');




%% Step 3: Image Resizing
inputSize = [224 224];       %GoogLeNet expects 224x224 input images.

augTrain = augmentedImageDatastore(inputSize, imdsTrain);
augVal   = augmentedImageDatastore(inputSize, imdsVal);
augTest  = augmentedImageDatastore(inputSize, imdsTest);



%% Step 4: Define Training Options

options = trainingOptions('sgdm', ...
    'MaxEpochs', 20, ...
    'MiniBatchSize', 32, ...
    'InitialLearnRate', 1e-4, ...  
    'Momentum', 0.9, ...
    'ValidationData', augVal, ...
    'ValidationFrequency', 50, ...
    'Plots','training-progress', ...
    'Verbose', true);



%% Step 5: Train the Network

[trainedNet, trainInfo] = trainNetwork(augTrain, lgraph_1, options);



%% Step 6: Evaluate on Test Set
predLabels = classify(trainedNet, augTest);

testAccuracy = mean(predLabels == imdsTest.Labels) * 100;
disp(['Test Accuracy (Transfer Learning w/GoogLeNet) = ', num2str(testAccuracy,4), '%']);

% Confusion Matrix
figure('Name','Confusion Matrix - Transfer Learning GoogLeNet');
confusionchart(imdsTest.Labels, predLabels);
title('Transfer Learning (GoogLeNet) - Test Set Confusion Matrix');



%% Step 7: Analyze Results

% Check final training/validation accuracy from 'trainInfo'
finalTrainAccuracy = trainInfo.TrainingAccuracy(end);
finalValAccuracy   = trainInfo.ValidationAccuracy(end);
disp(['Final Training Accuracy:   ', num2str(finalTrainAccuracy,4),'%']);
disp(['Final Validation Accuracy: ', num2str(finalValAccuracy,4),'%']);



