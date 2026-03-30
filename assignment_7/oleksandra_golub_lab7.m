close all
clear all

net = alexnet;
sz = net.Layers(1).InputSize;
analyzeNetwork(net)

%% cut layers
layersTransfer = net.Layers(1:end-3); % <--- TBD
% layersTransfer = freezeWeights(layersTransfer); % <--- TBD
% bisogna decidere il punto di taglio migliore 
% oppure decido solo una parte, solo prima tredici, solo..
% per adesso è tutto congelato 

layersTransfer(1:15) = freezeWeights(layersTransfer(1:15)); 
% primi 15 layer congelati, gli ultimi rimangono allenabili

%% replace layers
numClasses = 10;
layers = [layersTransfer 
    fullyConnectedLayer(numClasses, 'WeightLearnRateFactor', 20, ...
    'BiasLearnRateFactor', 20);
    softmaxLayer
    classificationLayer];
%analyzeNetwork(net)

% posso mettere learning rate diversi per diversi layers
% cosi alcuni si aggiornano piu velocemente e altri meno velocemente


% adesso bisogna preparare i dati per la nostra rete
% non devono essere sovrapposti i batch 
% rete fa forword in avanti e backword con i gradienti 
%% preparazione dati 
path_db = 'C:\Users\aleqs\Desktop\Visual\lab5\simplicityDB\image.orig\';
imds = imageDatastore(path_db);
% assegma in automanico label a cartella di classe 1, 2, 3, 4...
% ma nel nostro file non esiste, bisogna farlo manualmente

% si circa su file
% sulla base del npme, ssi ordina
labels = [];
for ii=1:size(imds.Files, 1)
    name = imds.Files{ii, 1};
    [p,n,ex] = fileparts(name);
    class = floor(str2double(n)/100);
    labels = [labels; class];
end

labels = categorical(labels);
imds = imageDatastore(path_db, 'labels', labels);


%% divisione train-val-test
% train 70% - test 30%
[imdsTrain, imdsTest] = splitEachLabel(imds, 0.7, 'randomized');
% dal train ricavo un vero validation set (80% train, 20% val)
[imdsTrain, imdsVal] = splitEachLabel(imdsTrain, 0.8, 'randomized');


%% data augmentation
% intervenire per trovare trasformazioni magari piu utili o piu corrette 
% sia geometriche che fotometriche
pixelRange = [-2 2];
imageAugmenter = imageDataAugmenter(...   % <--- TBD (fatto)
    'RandRotation', [-10 10], ...        % piccole rotazioni
    'RandXReflection', true, ...         % flip orizzontale
    'RandYReflection', true, ...         % flip verticale
    'RandXTranslation', pixelRange, ...
    'RandYTranslation', pixelRange, ...
    'RandXScale', [0.9 1.1], ...         % leggera variazione di scala
    'RandYScale', [0.9 1.1]); 

augimdsTrain = augmentedImageDatastore(sz(1:2), imdsTrain, 'DataAugmentation', imageAugmenter);
% lo si fa anche per test e val? si
% pero bisogna fare resize per immagini, non altro!
augimdsVal  = augmentedImageDatastore(sz(1:2), imdsVal);
augimdsTest = augmentedImageDatastore(sz(1:2), imdsTest);

%% configurazione fine-tuning
options = trainingOptions('sgdm', ...
    'MiniBatchSize', 32, ... % <--- TBD, al posto di 10
    'MaxEpochs', 17, ... % <--- TBD, al posto di 6
    'InitialLearnRate', 1e-4, ... % <--- TBD
    'LearnRateSchedule', 'piecewise', ... 
    'LearnRateDropFactor', 0.1, ...
    'LearnRateDropPeriod', 5, ...
    'Shuffle','every-epoch', ...
    'ValidationData', augimdsVal, ... % <--- TBD sistemato, al posto di augimdsTest
    'ValidationFrequency', 3, ...
    'Verbose',false, ...
    'ExecutionEnvironment', 'auto', ...
    'Plots','training-progress');

%% training vero e proprio
netTransfer = trainNetwork(augimdsTrain, layers, options);

%% test
tic
[lab_pred_te, scores] =  classify(netTransfer, augimdsTest);
toc

%% valutazione performance
acc = numel(find(lab_pred_te == imdsTest.Labels))/numel(lab_pred_te)
