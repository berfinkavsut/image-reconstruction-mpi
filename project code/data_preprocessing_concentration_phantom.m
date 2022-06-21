%% 1. Loading the required external functions
clear all
close all

%% 2. Download measurement and systemMatrix from http://media.tuhh.de/ibi/mdf/
filenameSM = 'systemMatrix.mdf';
filenameMeas = 'measurement.mdf';
websave(filenameSM,'http://media.tuhh.de/ibi/mdfv2/systemMatrix_V2.mdf')
websave(filenameMeas,'http://media.tuhh.de/ibi/mdfv2/measurement_V2.mdf')
  
%% 3. Loading the data
% For the System matrix (later named SM)
% to obtain infos on the file, use the command: infoSM = h5info(filename_SM);
% or read the format documentation
% read the data, saved as real numbers
S = h5read(filenameSM, '/measurement/data');
% reinterpret as complex numbers
S = complex(S.r,S.i);

% get rid of background frames
isBG = h5read(filenameSM, '/measurement/isBackgroundFrame');
S = S(isBG == 0,:,:,:);

% For the measurements
% read and convert the data as complex numbers
% note that these data contain 500 measurements
u = h5read(filenameMeas, '/measurement/data');
%u = squeeze(u(1,:,:,:) + 1i*u(2,:,:,:));
u = fft(cast(u,'double'));
u = u(1:(size(u,1)/2+1),:,:,:);

%% 4. Pre-process - Remove the frequencies which are lower than 30 kHz, as they are unreliable due to the 
anologue filter in the scanner

% generate frequency vector
numFreq = h5read(filenameMeas, '/acquisition/receiver/numSamplingPoints')/2+1;
rxBandwidth = h5read(filenameMeas, '/acquisition/receiver/bandwidth');
freq = linspace(0,1,numFreq) .* rxBandwidth;

% we supose that the same frequencies are measured on all channel for 
% the SM and the measurements. use only x/y receive channels
idxFreq = freq > 80e3;
S_truncated = S(:,idxFreq,1:2);
u_truncated = u(idxFreq,1:2,:);

% take calibration measurement from each coil 
% take transpose of them to have frequency components
% in columns, corresponding to each grid position 
SystemMatrix.S1 = transpose(S_truncated(:,:,1)); %764x6859
SystemMatrix.S2 = transpose(S_truncated(:,:,2)); %764x6859

% prepare the system matrix
% take real and imaginary parts of complex system matrices 
% concetanate the system matrices from two receive coils 
S1 = [real(SystemMatrix.S1);imag(SystemMatrix.S1)]; % 1528x6859
S2 = [real(SystemMatrix.S2);imag(SystemMatrix.S2)]; % 1528x6859
S = [S1;S2]; % 3056x6859

% take measurement data of resolution phantom from each coil 
u1 = u_truncated(:,1,:); u1 = reshape(u1,size(u1,1),size(u1,3)); %764x38000
u2 = u_truncated(:,2,:); u2 = reshape(u2,size(u2,1),size(u2,3)); %764x38000

% take average of measurements 
measurement.u1 = mean(u1,2); %764x1
measurement.u2 = mean(u2,2); %764x1

% prepare the measurement vector
% take real and imaginary parts of complex system matrices 
% concetanate the measurements from two receive coils 
u1_real = [real(measurement.u1);imag(measurement.u1)]; %1528x1
u2_real = [real(measurement.u2);imag(measurement.u2)]; %1528x1
u = [u1_real;u2_real]; %3056x1
save('SM_website.mat','S');
save('meas_website.mat','u');

% %% 5. Merge frequency and receive channel dimensions
% S_truncated = reshape(S_truncated, size(S_truncated,1), size(S_truncated,2)*size(S_truncated,3));
% u_truncated = reshape(u_truncated, size(u_truncated,1)*size(u_truncated,2), size(u_truncated,3));
% 
% %% 6. Averaged the measurement used for the reconstruction over all temporal frames
% u_mean_truncated = mean(u_truncated,2);
% 
% %% 7. Make two simple reconstructions
% % a normalized regularized kaczmarz approach
% c_normReguArt = kaczmarzReg(S_truncated(:,:),...
% u_mean_truncated(:),...
% 1,1*10^-6,0,1,1);
% 
% % and an regularized pseudoinverse approach
% [U,Sigma,V] = svd(S_truncated(:,:).','econ');
% Sigma2 = diag(Sigma);
% c_pseudoInverse = pseudoinverse(U,Sigma2,V,u_mean_truncated,5*10^2,1,1);
% 
% %% 8. Display an image
% % read the original size of an image
% number_Position = h5read(filenameSM, '/calibration/size');
% 
% figure
% subplot(1,2,1)
% imagesc(real(reshape(c_normReguArt(:),number_Position(1),number_Position(2))));
% colormap(gray); axis square
% title({'Regularized and modified ART - 3 channels';'1 iterations / lambda = 10^{-6} / real part'})
