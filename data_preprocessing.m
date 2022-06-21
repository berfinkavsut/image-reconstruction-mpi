%% Loading the required external functions
clear all
close all

%% System Matrix 
% load system matrix 
filenameSM = 'calibration2.mdf';
filenameMeas = 'resolutionPhantom2.mdf';
number_Position = h5read(filenameSM, '/calibration/size'); %[19;19;19]

% read the data, saved as real numbers
S = h5read(filenameSM, '/measurement/data');

% reinterpret as complex numbers
S = complex(S.r,S.i);

% get rid of background frames
isBG = h5read(filenameSM, '/measurement/isBackgroundFrame');
S = S(isBG == 0,:,:,:); 

% size of S : 6859x817x3
% # of grid positions: 6859 = 19x19x19 
% # of frequency components : 817
% # of receive coils : 3
% pre-process 
% remove the frequencies which are lower than 30 kHz, 
% as they are unreliable due to the anologue filter in the scanner
% generate frequency vector
numFreq = h5read(filenameMeas, '/acquisition/receiver/numSamplingPoints')/2+1;
rxBandwidth = h5read(filenameMeas, '/acquisition/receiver/bandwidth');
freq = linspace(0,1,numFreq) .* rxBandwidth; %1x817

% we suppose that the same frequencies are measured on all channel for 
% the SM and the measurements
% use only x/y receive channels
idxFreq = freq > 80e3;
S_truncated = S(:,idxFreq,1:2); %6859x764x2

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

%take calibration measurement for each slice and gather them in S_slices 
N = 19;
grid_pos_no = N*N; %NxNxN 3D image, one slice is NxN 2D image
patch_no = N; 
for k = 1:patch_no

 %take calibration measurements of grid positions for one slice 
 S_slice_truncated = S_truncated( (k-1)*grid_pos_no+1: k*grid_pos_no,:,:); %361x764x2 
 
 % take calibration measurement from two receive coils for each slice
 % similar to previous steps, for each slice, take transpose of system matrix
 % to have frequency components in columns, corresponding to each grid position 
 SystemMatrix.S1_slice = transpose(S_slice_truncated(:,:,1)); %764x361
 SystemMatrix.S2_slice = transpose(S_slice_truncated(:,:,2)); %764x361
 
 % prepare the system matrix for each slice
 % take real and imaginary parts of complex system matrices 
 % concetanate the system matrices from two receive coils 
 S1_slice = [real(SystemMatrix.S1_slice);imag(SystemMatrix.S1_slice)];
 S2_slice = [real(SystemMatrix.S2_slice);imag(SystemMatrix.S2_slice)];
 S_slices(:,:,k) = [S1_slice;S2_slice];
 
end 
 
%save system matrix 
save('SM_2D.mat','S');

%save system matrix for each slice
save('SM_2D_slice.mat','S_slices');

%% Measurement of Resolution Phantom 
% load measurement
filenameMeas = 'resolutionPhantom2.mdf';

% for the measurements
% read and convert the data as complex numbers
% note that these data contain 500 measurements
u = h5read(filenameMeas, '/measurement/data');
u = fft(cast(u,'double'));
u = u(1:(size(u,1)/2+1),:,:,:);

% use only x/y receive channels
u_truncated = u(idxFreq,1:2,:); %764x2x38000

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
u_final = [u1_real;u2_real]; %3056x1

%take measurement for each slice and gather them in u_slices 
patch_no = 19;
period_per_patch = 1000;
for k = 1:patch_no
 
 %take measurements of grid positions for one slice 
 u_slice = u(:,:,(k-1)*period_per_patch+1 : k*period_per_patch,:);%817x3x1000x2
 u_slice_truncated = u_slice(idxFreq,1:2,:); %764x2x2000
 
 % take calibration measurement from two receive coils for each slice
 u1_slice = u_slice_truncated(:,1,:); u1_slice = reshape(u1_slice,size(u1_slice,1),size(u1_slice,3)); %764x2000
 u2_slice = u_slice_truncated(:,2,:); u2_slice = reshape(u2_slice,size(u2_slice,1),size(u2_slice,3)); %764x2000
 
 % take linear combination of measurements for each slice
 measurement.u1_slice = mean(u1_slice,2); %764x1
 measurement.u2_slice = mean(u2_slice,2); %764x1
 
 % prepare the measurement vector for each slice
 u1_slice_real = [real(measurement.u1_slice);imag(measurement.u1_slice)]; %1528x1
 u2_slice_real = [real(measurement.u2_slice);imag(measurement.u2_slice)]; %1528x1
 u_slices(:,k) = [u1_slice_real; u2_slice_real]; %3056x1
end 

% save measurement vector 
u= u_final; 
save('meas_resolutionPhantom.mat','u');

% save measurement vector for each slice
save('meas_resolutionPhantom_slice.mat','u_slices');
