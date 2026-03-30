%%% Oleksandra Golub
%%% 856706

clear;
close all;

% lettura immagine
im = imread("C:\Users\aleqs\Desktop\Visual\lab3\RAWinput\RAWinput\IMG_1295ss.tiff");
figure(1), imshow(im), title('immagine Originale')

% conversione in double nel range [0 1]
im = im2double(im);

% AutoExposure (finto)
im = im./max(im(:)); % equivalente a fare max(max(im))
figure(1), clf, imshow(im), title('Immagine dopo AutoExposure')

%% demosaiking (demosaicing)
% configurazione del sensore (Bayer pattern)
% R G R G
% G B G B
% R G R G
% G B G B
% ----->>> TODO: implementare interpolazione
[H,W] = size(im);                 % dimensioni del RAW monocanale

R  = im(1:2:end, 1:2:end);        % campioni R
B  = im(2:2:end, 2:2:end);        % campioni B
G1 = im(1:2:end, 2:2:end);        % campioni G su (riga dispari, colonna pari)
G2 = im(2:2:end, 1:2:end);        % campioni G su (riga pari, colonna dispari)

% interpolazione bilineare a piena risoluzione
R = imresize(R,  [H W], 'bilinear');
B = imresize(B,  [H W], 'bilinear');
G = 0.5*( imresize(G1,[H W], 'bilinear') + imresize(G2,[H W], 'bilinear') );

im = cat(3,R,G,B);
figure(2), imshow(im), title('Immagine dopo Demosaicing')


%% AWB
% Gray World (GW)
S = size(im);
im = reshape(im, [], 3);

% Gray World fatto in classe (target 0.5 per canale) 
media = mean(im, 1);  % [meanR meanG meanB]
coeff_gw = [0.5 0.5 0.5] ./ media;  

im_gw = reshape( min(max(im*diag(coeff_gw),0),1), S );
figure(3), imshow(im_gw), title('AWB con Gray World')

% -->> TODO: implementare MaxRGB (WhitePoint, detto anche WhitePatch)
wp = prctile(im, 99);     % trova il "bianco quasi massimo" per ogni canale, mentre il 99° percentile serve per evitare pixel anomali
target = 0.95;            % target del bianco desiderato
coeff_max = target ./ wp; % calcola fattori di scala per portare i bianchi a target
im_maxrgb = reshape(min(max(im*diag(coeff_max),0),1), S); % bilancia i colori, limitando il range [0,1] 
figure(4), imshow(im_maxrgb), title('AWB con MaxRGB')

im = im_maxrgb; % altrimenti si userebbe im_gw di gray world 

%% color correction
% ----->>>TODO: matrice 24x3 da prendere da articolo 
% color checker e dividerli per 255
RGBr = [
    116, 81, 67;      % 1. dark skin
    199, 147, 129;    % 2. light skin
    91, 122, 156;     % 3. blue sky
    90, 108, 64;      % 4. foliage
    130, 128, 176;    % 5. blue flower
    92, 190, 172;     % 6. bluish green
    224, 124, 47;     % 7. orange
    68, 91, 170;      % 8. purplish blue
    198, 82, 97;      % 9. moderate red
    94, 58, 106;      % 10. purple
    159, 189, 63;     % 11. yellow green
    230, 162, 39;     % 12. orange yellow
    35, 63, 147;      % 13. blue
    67, 149, 74;      % 14. green
    180, 49, 57;      % 15. red
    238, 198, 20;     % 16. yellow
    193, 84, 151;     % 17. magenta
    0, 136, 170;      % 18. cyan
    245, 245, 243;    % 19. white 9.5 (.05 D)
    200, 202, 202;    % 20. neutral 8 (.23 D)
    161, 163, 163;    % 21. neutral 6.5 (.44 D)
    121, 121, 122;    % 22. neutral 5 (.70 D)
    82, 84, 86;       % 23. neutral 3.5 (1.05 D)
    49, 49, 51        % 24. black 2 (1.5 D)
] / 255.0; 

RGBc = []; % matrice 24x3 dei valori RGB camera della color checker

 %im_crop = imcrop(im);
 %im_crop = imresize(im_crop,5);
 %figure(5), imshow(im_crop)
 %[x,y] = ginput(24);
 %save coordinateCC x y
 %imwrite(im2double(im_crop),'crop_del_cc.png');

load coordinateCC.mat
im_crop = im2double(imread('crop_del_cc.png'));

x = round(x);
y = round(y);
R = 2; % dimensione intorno su cui fare la media
for ii=1:24
    crop = im_crop(y(ii)-R:y(ii)+R, x(ii)-R:x(ii)+R, :);
    crop = reshape(crop, [], 3);
    RGBc = [RGBc; mean(crop)];
end

M = inv(RGBc'*RGBc) * (RGBc'*RGBr); % pseudo-inversa

im = reshape(im, [], 3);
im = im * M;
im = reshape(im, S);

% clipping e gestione valori non finiti
im(~isfinite(im)) = 0;   % garantisce che ogni canale stia in [0,1]
im = min(max(im, 0), 1);  % taglia tutti i valori fuori dal range [0,1]

figure(5), imshow(im), title('Immagine dopo Color Correction')

%% enhancement
Sfactor = 1.1; % -----> TODO: modificare fattore per aumento saturazione

im_hsv_satur = rgb2hsv(im); % conversione in HSV

im_hsv_satur(:,:,2) = im_hsv_satur(:,:,2)*Sfactor; % modifica saturazione
im_hsv_satur(:,:,2) = min(im_hsv_satur(:,:,2), 1); % clipping saturazione

im_satur = hsv2rgb(im_hsv_satur); % conversione in RGB

figure(6), imshow(im_satur), title(sprintf('Immagine dopo Saturazione (x%.2f)', Sfactor))

%% contrasto con il gaussian filter
im_hsv_gauss = rgb2hsv(im); % conversione in HSV

im_hsv_gauss(:,:,2) = im_hsv_gauss(:,:,2)*Sfactor; % modifica saturazione
im_hsv_gauss(:,:,2) = min(im_hsv_gauss(:,:,2), 1); % clipping saturazione

% gamma correction adattiva con filtro gaussiano sul canale V
canaleV_gauss = im_hsv_gauss(:,:,3);

% si crea la maschera
mask_gauss = 1 - canaleV_gauss;

sigma = 5; % deviazione standard 

% smooth della maschera con filtro gaussiano
mask_gauss_smooth = imgaussfilt(mask_gauss, sigma);
mask_gauss_smooth = min(max(mask_gauss_smooth, 0), 1); 

% si calcola esponente gamma adattivo per ogni pixel
expo_gauss = 2 .^ ((128 - 255*mask_gauss_smooth) / 128);

% si applica gamma correction adattiva
nuovoCanaleV_gauss = canaleV_gauss .^ expo_gauss;
nuovoCanaleV_gauss = min(max(nuovoCanaleV_gauss, 0), 1); % stabilizzazione numerica

% si sovrascrive il canale V con quello corretto
im_hsv_gauss(:,:,3) = nuovoCanaleV_gauss;

im_gauss = hsv2rgb(im_hsv_gauss); % conversione in RGB

figure(7), imshow(im_gauss)
title(sprintf('Immagine dopo Saturazione (x%.2f) + Contrasto Adattivo Gaussiano (σ=%d)', Sfactor, sigma))

%% contrasto con il bilateral filter
im_hsv_bilat = rgb2hsv(im); % conversione in HSV

im_hsv_bilat(:,:,2) = im_hsv_bilat(:,:,2)*Sfactor; % modifica saturazione
im_hsv_bilat(:,:,2) = min(im_hsv_bilat(:,:,2), 1); % clipping saturazione

% gamma correction adattiva con filtro bilaterale sul canale V
canaleV_bilat = im_hsv_bilat(:,:,3);

% si crea la maschera
mask_bilat = 1 - canaleV_bilat;

degreeOfSmoothing = 0.01;  % sensibilità 
spatialSigma = 5;          % estensione spaziale 

% smooth della maschera con filtro bilaterale
mask_bilat_smooth = imbilatfilt(mask_bilat, degreeOfSmoothing, spatialSigma);
mask_bilat_smooth = min(max(mask_bilat_smooth, 0), 1);

% si calcola esponente gamma adattivo per ogni pixel
expo_bilat = 2 .^ ((128 - 255*mask_bilat_smooth) / 128);

% si applica gamma correction adattiva
nuovoCanaleV_bilat = canaleV_bilat .^ expo_bilat;
nuovoCanaleV_bilat = min(max(nuovoCanaleV_bilat, 0), 1); % stabilizzazione numerica

% si sovrascrive il canale V con quello corretto
im_hsv_bilat(:,:,3) = nuovoCanaleV_bilat;

im_bilat = hsv2rgb(im_hsv_bilat); % conversione in RGB

figure(8), imshow(im_bilat)
title(sprintf('Immagine dopo Saturazione (x%.2f) + Contrasto Adattivo Bilaterale (DoS=%.2f, σ=%d)', ...
              Sfactor, degreeOfSmoothing, spatialSigma))

%% compressione e salvataggio
imwrite(im2uint8(im_satur), 'immagine_raw_to_srgb_saturation6.jpg');
imwrite(im2uint8(im_gauss), 'immagine_raw_to_srgb_gaussian6.jpg');
imwrite(im2uint8(im_bilat), 'immagine_raw_to_srgb_bilateral6.jpg');