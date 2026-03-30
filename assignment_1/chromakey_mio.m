%%% Oleksandra Golub
%%% 856706

clear;
close all;

im1 = im2double( imread('godzilla_chromakey/godzilla_1.jpg'));
im2 = im2double( imread('godzilla_chromakey/godzilla_2.jpg'));

S1 = size(im1);   % dimensioni di im1 come [H, W, 3]
figure(1), imshow(im1), title('Immagine originale')


%%% selezione delle regioni da usare come modello per il green screen
% - opzione 1: interattiva
% crop = imcrop(im1);
% - opzione 2: dando le coordinate esplicitamente

r = 5;  % raggio metà-lato per ottenere crop 

% crop 1: [454.5, 351.5]
x1 = round(454.5); 
y1 = round(351.5);
crop1 = im1(y1-r:y1+r, x1-r:x1+r, :);
figure(2), imshow(crop1), title('Crop 1 [454.5,351.5]')

% crop 2: angolo alto-sinistra (usato in classe)
crop2 = im1(1:10, 1:10, :);
figure(3), imshow(crop2), title('Crop 2 [1..10,1..10]')

% crop 3: [416.5, 523.5]
x3 = round(416.5); 
y3 = round(523.5);
crop3 = im1(y3-r:y3+r, x3-r:x3+r, :);
figure(4), imshow(crop3), title('Crop 3 [416.5,523.5]')

% crop 4: [57.5, 305.5] 
x4 = round(57.5);  
y4 = round(305.5);
crop4 = im1(y4-r:y4+r, x4-r:x4+r, :);
figure(5), imshow(crop4), title('Crop 4 [57.5,305.5]')


%%% si usa lo spazio YCbCr di cui usiamo solo Cb,Cr 
% conversione dell'immagine in YCbCr
im1_ycc = rgb2ycbcr(im1); 
X = reshape(im1_ycc(:, :, 2:3), [], 2);    % matrice N×2 (N=H·W)
N = size(X, 1);     % numero totale di pixel

% conversione dei crop in YCbCr
c1_ycc = rgb2ycbcr(crop1);
c2_ycc = rgb2ycbcr(crop2);
c3_ycc = rgb2ycbcr(crop3);
c4_ycc = rgb2ycbcr(crop4);

% si estraggono i canali Cb e Cr dei 4 crop
m1 = reshape(c1_ycc(:,:,2:3), [], 2);
m2 = reshape(c2_ycc(:,:,2:3), [], 2);
m3 = reshape(c3_ycc(:,:,2:3), [], 2);
m4 = reshape(c4_ycc(:,:,2:3), [], 2);

% modello combinato (set di verdi in CbCr) dai 4 crop
modello = unique([m1; m2; m3; m4], 'rows');  % concatena tutti i campioni e rimuove duplicati

% soglia fissa T
T = 0.045;

%%% calcolo della similarità tra i pixel dell'immagine e quelli del modello
% con ciclo for
M_for = false(N,1);
tic
for ii = 1:N
    d = bsxfun(@minus, modello, X(ii,:));   % differenza con tutti i campioni del modello
    d = sqrt(sum(d.^2, 2));    % euclidean distance      
    M_for(ii) = (min(d) > T);   % se più lontano di T dal verde, allora si ha foreground
end
tempoFor = toc;

% vettorializzata
tic
dmin = min(pdist2(X, modello), [], 2); % distanza minima per ogni pixel da tutti i campioni     
M_vec = dmin > T;  % la maschera logica N×1 con true = foreground                   
tempoVec = toc;

% stampa dei tempi per confronto tra le due versioni
[tempoFor tempoVec]

%%% ricostruzione della maschera e dell'immagine
M2 = reshape(M_vec, S1(1), S1(2)); % si rimappa la maschera da N×1 a [H×W]        

% visualizzazione
figure(6), clf
subplot(1,3,1), imshow(im1), title('Immagine originale')
subplot(1,3,2), imshow(M2),  title('Maschera (non-green)')
subplot(1,3,3), imshow(im2), title('Background')

%%% creazione dell'immagine combinata
M3 = repmat(M2, 1, 1, 3);    % si replica la maschera su 3 canali per poterla applicare ad immagini RGB            
im_3 = im1 .* M3 + im2 .* (1-M3);  % la maschera RGB ottenuta
figure(7), imshow(im_3), title('Risultato finale (YCbCr, T = 0.045)')
