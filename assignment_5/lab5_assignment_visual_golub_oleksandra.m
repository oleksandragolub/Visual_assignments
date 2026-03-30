close all
clear all


disp('creazione griglia')
pointPositions = [];
featStep = 15; % TBD, al posto di 30 originali
imsize = 400; % TBD, al posto di 500 originali

tic
for ii = featStep:featStep:imsize-featStep
    for jj = featStep:featStep:imsize-featStep
        pointPositions=[pointPositions; ii jj];
    end
end
toc

%% estrazione features
disp('estrazione features')

Nim4training = 70; % TBD, fix nel caso di necessità del validation set (restato uguale a valore originale)
features = [];
labels = [];

tic
for class=0:9
    for nimage=0:Nim4training-1
        im=im2double(imread(['C:\Users\aleqs\Desktop\Visual\lab5\simplicityDB\image.orig\' ...
            num2str(100*class+nimage) '.jpg']));
        % TBD: provare a fare crop invece che resize (stretch)
        % pero ho usato sia crop che resize (evitando deformazione) 
        % per ottenere migliori prestazioni
        [h, w, ~] = size(im); % si legge altezza e larghezza originali
        side = min(h, w); % si ottiene il lato del quadrato massimo contenuto nell'immagine
        % si prendono le coordinate del crop centrale
        rowStart = floor((h - side)/2) + 1; % inizio del ritaglio verticale centrato
        colStart = floor((w - side)/2) + 1; % inizio del ritaglio orizzontale centrato

        im = im(rowStart:rowStart+side-1, colStart:colStart+side-1, :); % ritaglio centrale quadrato 
        im = imresize(im, [imsize imsize]); % quadrato viene ridimensionato a imsize × imsize, evitando stretch
        
        im = rgb2gray(im);
        [imfeatures,dontcare] = extractFeatures(im, pointPositions, 'Method','SURF');
        features = [features; imfeatures];
        labels = [labels; repmat(class,size(imfeatures,1),1) repmat(nimage,size(imfeatures,1),1)];
    end
end
toc

%% creazione del vocabolario
disp('kmeans')
K = 300; % TBD, al posto di 100 originali
tic
[IDX, C] = kmeans(features, K);
toc

%% istogrammi BOW training
disp('rappresentazione BOW training')
BOW_tr = [];
labels_tr = [];

tic
for class=0:9
    for nimage=0:Nim4training-1
        % si trovano tutti gli indici delle feature SURF estratte che appartengono all'immagine
        u=find(labels(:,1)==class & labels(:,2)==nimage); 
        % si prendono i visual words assegnati alle feature di quell'immagine tramite k-means
        imfeaturesIDX = IDX(u); 
        H = hist(imfeaturesIDX, 1:K); % si costruisce l'istogramma
        H = H./sum(H); % normalizzazione L1
        BOW_tr = [BOW_tr; H]; % si fa un'istogramma come vettore di feature dell'immagine
        labels_tr = [labels_tr; class]; % si salva la classe corretta dell'immagine
    end
end

N = size(BOW_tr, 1); % dice quante immagini ci sono nel training
% document frequency per word (in quante immagini compare la word in esame)
df = sum(BOW_tr > 0, 1);  
% inverse document frequency (serve a penalizzare le parole troppo frequenti e valorizzare quelle più informative)
idf = log(N ./ (df + 1));  

BOW_tr = BOW_tr .* repmat(idf, N, 1);  % si applica il peso TF-IDF agli istogrammi
% riduce l'effetto dei bin dominanti e stabilizza la distribuzione
BOW_tr = sign(BOW_tr) .* sqrt(abs(BOW_tr)); % power normalization
% normalizza la lunghezza dei vettori, rendendo le distanze più significative
BOW_tr = BOW_tr ./ repmat(sqrt(sum(BOW_tr.^2, 2)), 1, K); % normalizzazione L2 
toc


%% classificatore
% input: BOW_tr e labels_tr
% TBD

disp('training SVM con kernel RBF');
Mdl = fitcecoc(BOW_tr, labels_tr, ...
               'Learners', templateSVM('KernelFunction', 'rbf', ...
                                       'KernelScale', 'auto', ...
                                       'BoxConstraint', 10));

%% istogrammi BOW test
disp('rappresentazione BOW test')
BOW_te = [];
labels_te = [];

tic
for class=0:9
    for nimage=Nim4training:99
        im=im2double(imread(['C:\Users\aleqs\Desktop\Visual\lab5\simplicityDB\image.orig\' ...
            num2str(100*class+nimage) '.jpg']));
        % TBD: provare a fare crop invece che resize (stretch)
        % pero ho usato sia crop che resize per ottenere migliori
        % prestazioni
        [h, w, ~] = size(im);
        side = min(h, w);
        rowStart = floor((h - side)/2) + 1;
        colStart = floor((w - side)/2) + 1;
        im = im(rowStart:rowStart+side-1, colStart:colStart+side-1, :);
        im = imresize(im, [imsize imsize]);
        
        im = rgb2gray(im);
        [imfeatures,dontcare] = extractFeatures(im, pointPositions, 'Method','SURF');
        %%%
        D = pdist2(imfeatures, C);
        [dontcare, words]=min(D,[],2);
        %%%
        H = hist(words, 1:K);
        H = H./sum(H);

        H = H .* idf; % applica stesso IDF del training
        H = sign(H) .* sqrt(abs(H)); % power + L2 norm
        H = H ./ norm(H);
        
        BOW_te = [BOW_te; H];
        labels_te = [labels_te; class];
    end
end
toc

%% classificazione del test set
% TBD! aggiornare con il vostro vero classificatore (fatto)
disp('classificazione test set')
tic
predicted_class = predict(Mdl, BOW_te);
toc

%% misurazione performance
CM = confusionmat(labels_te, predicted_class);
CM = CM./repmat(sum(CM,2),1,size(CM,2));
CM
accuracy = mean(diag(CM))