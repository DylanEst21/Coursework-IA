
%% Part iii): Transfer Learning with AlexNet (Experiment 2)
% We have a layer graph "layers_1" exported from
% Deep Network Designer, containing the final layers for 11 classes.

clear; clc; close all;

%% Step 1: Load the modified network structure

if exist('modified_alexnet_layers.mat', 'file')
    load('modified_alexnet_layers.mat', 'layers_1');
else
    error('Modified network file not found.');
end
     



%% Step 2: Create Training, Validation and Test Datastrores
rootFolder = 'Food-11';

% Define the sub-folders for training, validation, and evaluation
trainFolder = fullfile(rootFolder, 'training');
valFolder   = fullfile(rootFolder, 'validation');
testFolder  = fullfile(rootFolder, 'evaluation');


% Create ImageDatastore, each datastore will automatically assign labels based on folder names
imdsTrain = imageDatastore(trainFolder, ...
    'IncludeSubfolders', true, ...
    'LabelSource', 'foldernames');

imdsVal = imageDatastore(valFolder, ...
    'IncludeSubfolders', true, ...
    'LabelSource', 'foldernames');

imdsTest = imageDatastore(testFolder, ...
    'IncludeSubfolders', true, ...
    'LabelSource', 'foldernames');



%% Step 3: Image Resizing

% AlexNet expects 227x227 input
inputSize = [227 227]; 


% Create augmentedImageDatastore to resize each image
augTrain = augmentedImageDatastore(inputSize, imdsTrain);
augVal   = augmentedImageDatastore(inputSize, imdsVal);
augTest  = augmentedImageDatastore(inputSize, imdsTest);



%% Step 4: Define Training Options

options = trainingOptions('sgdm', ...
    'MaxEpochs',       20, ...        
    'MiniBatchSize',    32, ...       
    'InitialLearnRate', 1e-4, ...     
    'Momentum',         0.9, ...
    'ValidationData',   augVal, ...
    'ValidationFrequency', 50, ...    %how often to check val accuracy
    'Plots','training-progress', ...
    'Verbose', true);



%% Step 5: Train the Network

[trainedNet, trainInfo] = trainNetwork(augTrain, layers_1, options);    % The function 'trainNetwork' can directly take a layerGraph




%% Step 6: Evaluate on Test Set
predLabels = classify(trainedNet, augTest);

testAccuracy = mean(predLabels == imdsTest.Labels) * 100;
disp(['Test Accuracy (Transfer Learning) = ', num2str(testAccuracy,4),'%']);

% Confusion matrix
figure('Name','Confusion Matrix - Transfer Learning AlexNet');
confusionchart(imdsTest.Labels, predLabels);
title('Transfer Learning (AlexNet) - Test Set Confusion Matrix');



%% Step 7: Analyze Results

% Check final training/validation accuracy from 'trainInfo'
finalTrainAccuracy = trainInfo.TrainingAccuracy(end);
finalValAccuracy   = trainInfo.ValidationAccuracy(end);
disp(['Final Training Accuracy:   ', num2str(finalTrainAccuracy,4),'%']);
disp(['Final Validation Accuracy: ', num2str(finalValAccuracy,4),'%']);






