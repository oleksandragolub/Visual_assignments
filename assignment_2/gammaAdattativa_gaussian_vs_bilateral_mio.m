%%% Oleksandra Golub
%%% 856706

clear;
close all;

im = imread("archivio_E03/underexposed.jpg");

figure(1)
imshow(im), title('immagine originale')

im = im2double(im);

%% gamma correction globale

% si applica una correzione gamma per schiarire immagine
gamma=0.4;
im1 = im.^gamma;

% si applica una correzione gamma per scurire immagine
gamma=4;
im2 = im.^gamma;

% si applica una correzione gamma diversa a ogni pixel
gamma = 4*rand(size(im,1),size(im,2));   % a causa di 4*, gamma varia tra 0 e 4
im3 = im.^gamma;

figure(2)
subplot(1,3,1), imshow(im1), title('immagine schiarita')
subplot(1,3,2), imshow(im2), title('immagine scurita')
subplot(1,3,3), imshow(im3), title('immagine scurita/schiarita casualmente')

%% gamma correction adattativa
% converto in ycbcr
Ycbcr = rgb2ycbcr(im);
Ycbcr_gauss = Ycbcr;
Ycbcr_b = Ycbcr;

% estraggo il canale Y
canaleY_gauss = Ycbcr_gauss(:,:,1);
canaleY_b  = Ycbcr_b(:,:,1);

% TBD: faccio le operazioni che devo fare sul canale Y

% calcolo la maschera
mask_g = 1 - canaleY_gauss;
mask_b = 1 - canaleY_b;

mask_gauss = imgaussfilt(mask_g, 5); % maschera invertita, deviazione standard della gaussiana (sigma)
mask_gauss = min(max(mask_gauss,0),1);  % serve per evitare che la maschera esca dai limiti di luminosità validi (0–1)

figure(3), 
subplot(1,2,1), imshow(mask_g), title('mask (inverted Y)')
subplot(1,2,2), imshow(mask_gauss), title('mask imgaussfilt')

mask_bi = imbilatfilt(mask_b, 0.1, 5); % maschera invertita, degreeOfSmoothing, spatialSigma
mask_bi = min(max(mask_bi,0),1);  % serve per evitare che la maschera esca dai limiti di luminosità validi (0–1)

figure(4), 
subplot(1,2,1), imshow(mask_b), title('mask (inverted Y)')
subplot(1,2,2), imshow(mask_bi), title('mask imbilatfilt')

% esponente per pixel:  gamma(x) = 2^((128 - 255*mask)/128)
expo_bi = 2 .^ ((128 - 255*mask_bi) / 128);
expo_gauss = 2 .^ ((128 - 255*mask_gauss) / 128);

% nuovoCanaleY: Y_out = Y_in ^ expo
nuovoCanaleY_gauss = canaleY_gauss .^ expo_gauss;
nuovoCanaleY_b = canaleY_b .^ expo_bi;

% stabilizzazione numerica e clamp
nuovoCanaleY_gauss = min(max(nuovoCanaleY_gauss, 0), 1);
nuovoCanaleY_b = min(max(nuovoCanaleY_b, 0), 1);


% sovrascrivo il canale Y
Ycbcr_gauss(:,:,1) = nuovoCanaleY_gauss;
% riconverto in RGB
im_gauss = ycbcr2rgb(Ycbcr_gauss);

% sovrascrivo il canale Y
Ycbcr_b(:,:,1) = nuovoCanaleY_b;
% riconverto in RGB
im_b = ycbcr2rgb(Ycbcr_b);

figure(5), 
subplot(1,2,1), imshow(im_gauss), title('gaussiano')
subplot(1,2,2), imshow(im_b), title('bilaterale')