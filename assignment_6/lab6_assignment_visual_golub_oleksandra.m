close all
clear all

%% caricamento di AlexNet per analizzare la rete
net = alexnet; % si carica la rete neurale 
analyzeNetwork(net) % si apre il tool grafico

sz = net.Layers(1).InputSize; % si prendono le dimensioni dell'input richiesto dalla rete

indir  = 'C:\Users\aleqs\Desktop\Visual\lab6\simplicityDB\image.orig\';
Nim4tr = 70; % numero di immagini per classe usate per il training

% layers da testare: vicino input, metà rete, vicino output
layers_to_test = {'conv1','conv3','relu7'};

%% visualizzo le feature di conv1 per peppers.png
layer = 'conv1';
im = imread('peppers.png');
figure; imshow(im), title('immagine di input')

im = double(imresize(im, sz(1:2)));
feat = activations(net, im, layer); % si calcolano le attivazioni della rete al layer conv1

figure(1), clf
for ii = 1:25 % visualizzazione delle prime 25 feature map
    subplot(5, 5, ii)
    imagesc(feat(:,:,ii)), colormap gray, axis off
end

%% confronto performance per i 3 layer, con e senza normalizzazione
acc_noNorm = zeros(numel(layers_to_test),1);
acc_norm   = zeros(numel(layers_to_test),1);

for L = 1:numel(layers_to_test)
    layer = layers_to_test{L};
    disp('==============================')
    disp(['LAYER: ' layer])
    disp('==============================')

    % estrazione feature per questo layer
    [feat_tr, labels_tr, feat_te, labels_te] = ...
        extract_features_layer(net, layer, indir, Nim4tr, sz);

    %% classificazione 1-NN SENZA normalizzazione
    D = pdist2(feat_te, feat_tr);
    [~, idx_pred_te] = min(D, [], 2);
    lab_pred_te = labels_tr(idx_pred_te);
    acc_noNorm(L) = sum(lab_pred_te == labels_te) / numel(labels_te);

    %% classificazione 1-NN CON normalizzazione L2
    feat_tr_n = feat_tr ./ sqrt(sum(feat_tr.^2,2));
    feat_te_n = feat_te ./ sqrt(sum(feat_te.^2,2));

    Dn = pdist2(feat_te_n, feat_tr_n);
    [~, idx_pred_te_n] = min(Dn, [], 2);
    lab_pred_te_n = labels_tr(idx_pred_te_n);
    acc_norm(L) = sum(lab_pred_te_n == labels_te) / numel(labels_te);

    disp(['Accuracy senza norm: ' num2str(acc_noNorm(L))])
    disp(['Accuracy con norm: ' num2str(acc_norm(L))])
end

%% Tabella riassuntiva
T = table(layers_to_test', acc_noNorm, acc_norm, ...
          'VariableNames', {'Layer','Acc_noNorm','Acc_norm'});
disp(' ')
disp('RISULTATI FINALI:')
disp(T)
