%%% Oleksandra Golub
%%% 856706

clear;
close all;

%% lettura immagini
% SIFT originalmente progettato per immagini a grigio
boxImage = imread("C:\Users\aleqs\Desktop\Visual\lab4\immaginiObjectDetection\elephant.jpg");
sceneImage = imread("C:\Users\aleqs\Desktop\Visual\lab4\immaginiObjectDetection\clutteredDesk.jpg");

%% keypoint detection
boxPoints = detectSURFFeatures(boxImage);
scenePoints = detectSURFFeatures(sceneImage);

% visualizzazione dei primi 100 keypoint più forti dell'immagine oggetto
figure(1);
imshow(boxImage), hold on
plot(selectStrongest(boxPoints, 100)), 
title('Keypoints of Object Image')
hold off

% visualizzazione dei primi 100 keypoint più forti dell'immagine scena
figure(2);
imshow(sceneImage), hold on
plot(selectStrongest(scenePoints, 100)),
title('Keypoints of Scene Image')
hold off

%% keypoint description
% estrazione delle feature per ogni keypoint
% 'Upright', true serve a disattivare la rotazione automatica dei descrittori
% dato che l'elefante nella scena non è ruotato, si può farlo per diminuire
% il rumore nei descrittori
[boxFeature, boxPoints] = extractFeatures(boxImage, boxPoints, 'Upright', true);
[sceneFeature, scenePoints] = extractFeatures(sceneImage, scenePoints, 'Upright', true);

%% features matching
% matching tra le feature dell'elefante e della scena
% 'MatchThreshold', 70 è il livello di tolleranza per accettare un match in base alla distanza
% 'MaxRatio', 0.8 serve a capire quale match tra diverse coppie testate è più affidabile
% 'Unique', true dice di non riutilizzare lo stesso punto per più match
% 'Method', 'Exhaustive'imposta il metodo di ricerca che confronta tutti contro tutti
boxPairs = matchFeatures(boxFeature, sceneFeature, ...
     'MatchThreshold', 70, 'MaxRatio', 0.8, 'Unique', true, 'Method', 'Exhaustive');

matchedBoxPoints = boxPoints(boxPairs(:,1), :);
matchedScenePoints = scenePoints(boxPairs(:,2), :);

% visualizzazione dei match iniziali
figure(3);
showMatchedFeatures(boxImage, sceneImage, matchedBoxPoints, matchedScenePoints, ...
    'montage');
title('Initial feature matches');

%% geometric consistency check detto anche RANSAC
% stima della trasformazione geometrica per filtrare i match corretti
% cioè pulisce i match da quelli che non ci servono e lascia solo quelli
% utili

% 'projective' è adatta a oggetti con prospettiva (inclina/ruota piano dell'oggetto)
% mentre a 'affine' basta se ci sono solo rotazione/traslazione senza prospettiva
% il primo sembra più fedele alla situazione di una scena con tanti oggetti
% quindi con una certa profondità
[tform, inlierBoxPoints, inlierScenePoints] = ...
    estimateGeometricTransform(matchedBoxPoints, matchedScenePoints, 'projective', ...
    'MaxDistance', 3, 'Confidence', 99.5, 'MaxNumTrials', 5000);

% con meno di 8 inlier si degrada facilmente, percio' meglio rifare il processo migliorando i parametri
if inlierBoxPoints.Count < 8   
    [tform, inlierBoxPoints, inlierScenePoints] = estimateGeometricTransform( ...
        matchedBoxPoints, matchedScenePoints, 'projective', ...
        'MaxDistance', 3.5, 'Confidence', 99.5, 'MaxNumTrials', 7000);
end

% visualizzazione degli inlier dopo RANSAC
figure(4);
showMatchedFeatures(boxImage, sceneImage, inlierBoxPoints, inlierScenePoints, ...
    'montage');
title('Matched features dopo RANSAC');

%% bounding box drawing 
% definizione dei 4 angoli dell'immagine dell'elefante
boxPoly = [1, 1;
    size(boxImage, 2), 1;
    size(boxImage, 2), size(boxImage, 1);
    1, size(boxImage, 1);
    1, 1];

% trasformazione del rettangolo nella scena
newboxPoly = transformPointsForward(tform, boxPoly);

figure(5);
imshow(sceneImage), hold on
line(newboxPoly(:, 1), newboxPoly(:, 2), 'Color', 'y', 'LineWidth', 2)
title('Bounding Box attorno oggetto')
hold off

%% contorno preciso dell'elefante (almeno 10 punti)
figure(6);
imshow(boxImage, 'InitialMagnification', 100); 
title('Seleziona almeno 10 punti lungo il contorno dell''elefante (doppio click per finire)');

% si apre il file di coordinate che ho creato
if isfile('coordinatesElefante.mat')
    load('coordinatesElefante.mat');  
else
    % altrimenti si scelgono manualmente nuove coordinate e si salvono
    [x, y] = ginput();  

    % chiusura del contorno
    x = [x; x(1)];
    y = [y; y(1)];

    % salvataggio dei punti
    save('coordinatesElefante.mat', 'x', 'y');
end

% chiusura del contorno
x = [x; x(1)];
y = [y; y(1)];

% drawing del contorno sull'immagine originale
hold on
line(x, y, 'Color', 'r', 'LineWidth', 2)
plot(x(1:end-1), y(1:end-1), 'ro', 'MarkerSize', 8, 'LineWidth', 2)
title(sprintf('Contorno dell''elefante (%d punti)', length(x)-1))
hold off

%% trasformazione del contorno nella scena
% si applica la trasformazione geometrica ai punti del contorno
newElephantContour = transformPointsForward(tform, [x, y]);

offset = mean(inlierScenePoints.Location - transformPointsForward(tform, inlierBoxPoints.Location));
newElephantContour = newElephantContour + offset;

% visualizzazione finale del contorno trasformato sulla scena
figure(7);
imshow(sceneImage), hold on
line(newElephantContour(:, 1), newElephantContour(:, 2), 'Color', 'r', 'LineWidth', 3)
plot(newElephantContour(1:end-1, 1), newElephantContour(1:end-1, 2), ...
    'yo', 'MarkerSize', 8, 'MarkerFaceColor', 'y')
title(sprintf('Elephant Detection con contorno preciso (%d punti)', length(x)-1))
hold off

% display delle statistiche
fprintf('Numero totale di match: %d\n', size(boxPairs, 1));
fprintf('Numero di inlier dopo RANSAC: %d\n', size(inlierBoxPoints, 1));
fprintf('Numero di punti nel contorno: %d\n', length(x)-1);