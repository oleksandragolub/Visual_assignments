% serve per estrarre le feature dal layer indicato per training e test
function [feat_tr, labels_tr, feat_te, labels_te] = extract_features_layer(net, layer, indir, Nim4tr, sz)
% net è la rete (alexnet)
% layer è uno dei tre layer testati ('conv1', 'conv3', 'relu7')
% indir è la cartella delle immagini simplicityDB\image.orig\
% Nim4tr è il numero di immagini per classe per il training (nel nostro caso sono 70)
% sz è la dimensione input rete 

feat_tr = []; % conterrà le feature del training set
labels_tr = []; % conterrà le etichette corrispondenti
feat_te = []; % conterrà le feature del test set
labels_te = []; % conterrà le etichette corrispondenti per il test set

%% estrazione features sul training set
for class = 0:9
    for nimage = 0:Nim4tr-1
        disp(['TRAIN - class ' num2str(class) ' img ' num2str(nimage)])
        im = double(imread([indir num2str(class*100 + nimage) '.jpg']));
        im = imresize(im, sz(1:2));

        % vettore riga 1×D
        feat_tmp = activations(net, im, layer, 'OutputAs','rows');

        feat_tr   = [feat_tr; feat_tmp];
        labels_tr = [labels_tr; class];
    end
end

%% estrazione features sul test set
for class = 0:9
    for nimage = Nim4tr:99
        disp(['TEST  - class ' num2str(class) ' img ' num2str(nimage)])
        im = double(imread([indir num2str(class*100 + nimage) '.jpg']));
        im = imresize(im, sz(1:2));

        feat_tmp = activations(net, im, layer, 'OutputAs','rows');

        feat_te   = [feat_te; feat_tmp];
        labels_te = [labels_te; class];
    end
end
end
