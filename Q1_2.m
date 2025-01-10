
%% Part ii): AlexNet Feature Extraction & Classification (Experiment 1)


%% Step 1: Unzip, load and split the new images as an image datastore. 

%unzip('archive.zip');     --> Already unzipped



%% Step 2: LOAD PRETRAINED ALEXNET AND DEFINE DATA PATHS

% Clear workspace and command window for cleanliness
clear; clc; close all;

% Load the pretrained AlexNet
net = alexnet;

% Specify the root folder where "Food-11" dataset is located
rootFolder = 'Food-11';

% Define the sub-folders for training, validation, and evaluation
trainFolder = fullfile(rootFolder, 'training');
valFolder   = fullfile(rootFolder, 'validation');
testFolder  = fullfile(rootFolder, 'evaluation');

%% Step 3: CREATE IMAGE DATASTORES FOR TRAIN, VALIDATION, TEST 
% --> CHOOSE DATA SPLIT APPROACH !!!


% =========================================
% Approach A: USE ALL IMAGES IN EACH FOLDER
% =========================================

% --> This approach simply takes all images in training, validation, and evaluation
% Use all images in each folder (training, validation, evaluation).


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






%{
% ======================================================
% Approach B (COMMENTED OUT): LIMIT EACH CLASS TO 30/5/5
% ======================================================
% In this approach, we randomly select 30 images per class from 'training',
% 5 images per class from 'validation', and 5 images per class from 'evaluation',
% ignoring any surplus. Limit each class to 30/5/5 images from training, validation, evaluation, respectively.


numTrainPerClass = 30;
numValPerClass = 5;
numTestPerClass = 5;


% --- Training (30/class) ---
imdsFullTrain = imageDatastore(trainFolder, ...
    'IncludeSubfolders', true, ...
    'LabelSource','foldernames');
tblTrain = countEachLabel(imdsFullTrain);  % Table --> Count images per label
trainFiles = {};
trainLabels = {};

for iClass = 1:numel(tblTrain.Label)     %numel() renvoie le nombre totald'éléments
    thisLabel = tblTrain.Label(iClass);
    idx = find(imdsFullTrain.Labels == thisLabel);  %Sélection des indices des images appartenant à cette classe
    idx = idx(randperm(numel(idx)));  % shuffle
    nToKeep = min(numTrainPerClass, numel(idx));    %if less than 30 exist, keep all
                    %min() renvoie la plus petite valeur parmi les 2 arguments
    idx = idx(1:nToKeep);   %extrait les premiers nToKeep indices après avoir mélangé.
    trainFiles = [trainFiles; imdsFullTrain.Files(idx)];
    trainLabels = [trainLabels; imdsFullTrain.Labels(idx)];
end
imdsTrain = imageDatastore(trainFiles, 'Labels', trainLabels);


% --- Validation (5/class) ---
imdsFullVal = imageDatastore(valFolder, ...
    'IncludeSubfolders', true, ...
    'LabelSource','foldernames');
tblVal = countEachLabel(imdsFullVal);
valFiles = {};
valLabels = {};

for iClass = 1:numel(tblVal.Label)
    thisLabel = tblVal.Label(iClass);
    idx = find(imdsFullVal.Labels == thisLabel);
    idx = idx(randperm(numel(idx)));
    nToKeep = min(numValPerClass, numel(idx));
    idx = idx(1:nToKeep);
    valFiles = [valFiles; imdsFullVal.Files(idx)];
    valLabels = [valLabels; imdsFullVal.Labels(idx)];
end
imdsVal = imageDatastore(valFiles, 'Labels', valLabels);


% --- Test (5/class) ---
imdsFullTest = imageDatastore(testFolder, ...
    'IncludeSubfolders', true, ...
    'LabelSource','foldernames');
tblTest = countEachLabel(imdsFullTest);
testFiles = {};
testLabels = {};

for iClass = 1:numel(tblTest.Label)
    thisLabel = tblTest.Label(iClass);
    idx = find(imdsFullTest.Labels == thisLabel);
    idx = idx(randperm(numel(idx)));
    nToKeep = min(numTestPerClass, numel(idx));
    idx = idx(1:nToKeep);
    testFiles = [testFiles; imdsFullTest.Files(idx)];
    testLabels = [testLabels; imdsFullTest.Labels(idx)];
end
imdsTest = imageDatastore(testFiles, 'Labels', testLabels);

% END OF SUBSET APPROACH



%}







% Display basic info (e.g., number of images, classes)
disp('================== DATASET INFO ==================');
disp(['Training set:   ', num2str(numel(imdsTrain.Files)), ' images']);
disp(['Validation set: ', num2str(numel(imdsVal.Files)),   ' images']);
disp(['Test set:       ', num2str(numel(imdsTest.Files)),  ' images']);
disp(['Number of classes: ', num2str(numel(unique(imdsTrain.Labels)))]);
disp('==================================================');


% You might want to verify each class has enough images in each subset:
% Vérifier les classes et les déséquilibres éventuels:
disp('Classes dans l’ensemble d’entraînement :');
disp(countEachLabel(imdsTrain));

disp('Classes dans l’ensemble de validation :');
disp(countEachLabel(imdsVal));

disp('Classes dans l’ensemble de test :');
disp(countEachLabel(imdsTest));



%{
Figure 3: Small grid of 16 randomly selected training images from the dataset, showcasing different food items for classification.
numTrainImages = numel(imdsTraining.Labels);  
idx = randperm(numTrainImages, 16);  

figure
for i = 1:16
    subplot(4, 4, i)   
    I = readimage(imdsTraining, idx(i));  
    imshow(I) 
    title(sprintf('Image %d', i))  
end
%}



%% Step 4: Image RESIZING AND EXTRACT FEATURES (FC7) FROM ALEXNET

inputSize = net.Layers(1).InputSize(1:2);   % AlexNet expects 227x227 input images


% Create augmentedImageDatastore to resize each image
augTrain = augmentedImageDatastore(inputSize, imdsTrain);
augVal   = augmentedImageDatastore(inputSize, imdsVal);
augTest  = augmentedImageDatastore(inputSize, imdsTest);

% Extract features from the 'fc7' layer of AlexNet (4096-dimensional)

featuresTrain = activations(net, augTrain, 'fc7', 'OutputAs','rows');
featuresVal   = activations(net, augVal,   'fc7', 'OutputAs','rows');
featuresTest  = activations(net, augTest,  'fc7', 'OutputAs','rows');


whos featuresTrain



%Extract the class labels from the training, validation and evaluation data.
%Retrieve the corresponding labels
labelsTrain = imdsTrain.Labels;
labelsVal   = imdsVal.Labels;
labelsTest  = imdsTest.Labels;



%% Step 5.1: TRAIN & EVALUATE SVM CLASSIFIER
disp('Training SVM classifier on extracted features...');

% Train a multiclass SVM using fitcecoc
svmModel = fitcecoc(featuresTrain, labelsTrain);


% Predict on Training data
predTrainSVM = predict(svmModel, featuresTrain);
accTrainSVM  = mean(predTrainSVM == labelsTrain)*100;

% Predict on Validation data
predValSVM = predict(svmModel, featuresVal);
accValSVM  = mean(predValSVM == labelsVal)*100;

% Predict on Test data
predTestSVM = predict(svmModel, featuresTest);
accTestSVM  = mean(predTestSVM == labelsTest)*100;

disp(['SVM Accuracy: Train=', num2str(accTrainSVM,4), ...
      '%, Val=', num2str(accValSVM,4), ...
      '%, Test=', num2str(accTestSVM,4),'%']);

%Plot Confusion Matrix for the Test set (SVM)
figure('Name','Confusion Matrix - SVM');
confusionchart(labelsTest, predTestSVM);
title('SVM Confusion Matrix (Test Set)');


%% Step 5.2: TRAIN & EVALUATE KNN CLASSIFIER
disp('Training KNN classifier on extracted features...');

% Choose the number of neighbors
kNeighbors = 5;

knnModel = fitcknn(featuresTrain, labelsTrain, 'NumNeighbors', kNeighbors);

% Predict on Training data
predTrainKNN = predict(knnModel, featuresTrain);
accTrainKNN  = mean(predTrainKNN == labelsTrain)*100;

% Predict on Validation data
predValKNN = predict(knnModel, featuresVal);
accValKNN  = mean(predValKNN == labelsVal)*100;

% Predict on Test data
predTestKNN = predict(knnModel, featuresTest);
accTestKNN  = mean(predTestKNN == labelsTest)*100;

disp(['KNN (k=',num2str(kNeighbors),') Accuracy: Train=', num2str(accTrainKNN,4), ...
      '%, Val=', num2str(accValKNN,4), ...
      '%, Test=', num2str(accTestKNN,4),'%']);

%Plot Confusion Matrix for the Test set (KNN)
figure('Name','Confusion Matrix - KNN');
confusionchart(labelsTest, predTestKNN);
title(['KNN Confusion Matrix (Test Set), k=', num2str(kNeighbors)]);



%% Step 5.3: TRAIN & EVALUATE A SHALLOW NEURAL NETWORK
disp('Training shallow Neural Network on extracted features...');

% Convert categorical labels to numeric indices
numClasses = numel(categories(labelsTrain));
YTrainIdx  = grp2idx(labelsTrain)';  % row vector; labelsTrain is converted to numeric indices using grp2idx
YTrainOneHot = full(ind2vec(YTrainIdx, numClasses)); %YTrainIdx is converted to a one-hot encoding matrix 
                                                     %using ind2vec, which is required for neural network training
% "ind2vec" requires the Neural Network Toolbox format

% --> This means each class is represented by a vector with a 1 at the position corresponding to the class index and 
% 0's elsewhere.


% We define a small feed-forward network with 2 hidden layers
hiddenLayerSizes = [100, 50];   %First hidden layer with 100 neurons and the second with 50 neurons.
netNN = patternnet(hiddenLayerSizes);   
    %patternnet() --> Creates a feed-forward neural network designed for classification tasks.

% We set training parameters
netNN.trainParam.epochs = 200;
netNN.trainParam.showWindow = true;   % set to false to hide training GUI


% Train the neural network
[netNN, tr] = train(netNN, featuresTrain', YTrainOneHot);
    % train() --> optimizes the weights and biaises of the network and returns they trained NN and 
    % a structure (tr) containing training details.


% Predict on Training data
predTrainNNRaw = netNN(featuresTrain');     %generates raw outputs (activation values) for each class (matrix)
[~, predTrainNNIdx] = max(predTrainNNRaw, [], 1);  %finds the class index with the highest score for each sample (predicted class)
predTrainNN = categorical(predTrainNNIdx, 1:numClasses, categories(labelsTrain));   %Converts the numeric indices (predNNIdx) back to categorical values, matching the original class labels
accTrainNN  = mean(predTrainNN' == labelsTrain)*100;

% Predict on Validation data
predValNNRaw = netNN(featuresVal');
[~, predValNNIdx] = max(predValNNRaw, [], 1);
predValNN = categorical(predValNNIdx, 1:numClasses, categories(labelsVal));
accValNN  = mean(predValNN' == labelsVal)*100;

% Predict on Test data
predTestNNRaw = netNN(featuresTest');
[~, predTestNNIdx] = max(predTestNNRaw, [], 1);
predTestNN = categorical(predTestNNIdx, 1:numClasses, categories(labelsTest));
accTestNN  = mean(predTestNN' == labelsTest)*100;

disp(['NN Accuracy: Train=', num2str(accTrainNN,4), ...
      '%, Val=', num2str(accValNN,4), ...
      '%, Test=', num2str(accTestNN,4),'%']);

%Plot Confusion Matrix for the Test set (NN)
figure('Name','Confusion Matrix - Neural Network');
confusionchart(labelsTest, predTestNN);
title('Neural Network Confusion Matrix (Test Set)');


%% Step 6: SUMMARY OF RESULTS
% -------------------------------------------------------
disp('==================== FINAL RESULTS ====================');
disp(['SVM:  Train=', num2str(accTrainSVM,4), '%,  Val=', ...
      num2str(accValSVM,4), '%,  Test=', num2str(accTestSVM,4),'%']);
disp(['KNN:  Train=', num2str(accTrainKNN,4), '%,  Val=', ...
      num2str(accValKNN,4), '%,  Test=', num2str(accTestKNN,4),'%']);
disp(['NN:   Train=', num2str(accTrainNN,4), '%,  Val=', ...
      num2str(accValNN,4), '%,  Test=', num2str(accTestNN,4),'%']);
disp('========================================================');




