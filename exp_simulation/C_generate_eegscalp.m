%clear all ;clc;
%% STEP 1: load Source EEG from file and import simulate signals from brainstorm
load SEEG
%%load from .mat file to workspace
% import S1 as 's'
% import S1_original as 'es'
% import S1_oringinal cortex as 'cs'
% SNR :0 15 20 25
% version 1
% V1 L: S1
% V1 R: S3 
% V2 L: S2
% V2 R: S4
% version 2
% S1: lateral occipital R
% S2: rostralmiddlefrontal L
% S3: rostralmiddlefrontal R
% S4: supramarginal L
S1=eeg_s{1};
S2=eeg_s{2};
S3=eeg_s{3};
S4=eeg_s{4};
source=[S1;S3;S2;S4];
% OPEN brainstorm and choose the file and right click to 'import the data
% matrix', then choose source and enter the sampling rate as '1000' 
% plz name simulate signals import from brainstorm as 's'
s.Value=source;
%% STEP 2:use brainstorm to simulate signals from scouts

% this step is manipulated in the brainstorm toolbox
% import s to S1 signals ,and use S1 to generate the signals from scouts
%% STEP 3: import cs ss from brainstorm
% name cs : simulated cortex source ,import from brainstorm
% name es : simulated EEG Signals ,import from brainstorm
% set parameters of EEG
num=60; % number of event stimulated module
efs=100; % EEG sample frequency 
itv=8;   %interval of each event
dt=41;   %duration time of ERP
snr=10 ;   %signal noisy ratio
%% process eeg signals:intercept ERP signals
eeg=es.F;
% add some noise for simulated signals
if snr~=0
    eeg=awgn(eeg,snr,'measured','dB');
end
%intercepting
chan=cs.GoodChannel;
channel=size(chan,2);  %good channels of eeg  
%eeg=eeg(chan,:); %±£¡ÙÀ˘”–channels
erp=cell(1,num);
eeg_f=[];
for i=1:num
    erp{i}=eeg(:,1+(i-1)*itv*efs:dt+(i-1)*itv*efs);
    eeg_f=[eeg_f erp{i}]; %eeg final 
end
% plot(eeg_f);
es_c=es;
es_c.F=eeg_f;
es_c.Time=0:1:dt*num-1;
% es.F=eeg_f;
% es.Time=0:1:dt*num-1;
%% process cortex source:intercept ERP signals
ca=cs.ImageGridAmp;    %cortex amplitude
ca_t=cell(1,num);      %trails of cortex amplitude 
ca_f=[];               %finale cortex amplitude
%intercepting
for i=1:num
    ca_t{i}=ca(:,1+(i-1)*itv*efs:dt+(i-1)*itv*efs);
    ca_f=[ca_f ca_t{i}];
end
%reload
cs_c=cs;
cs_c.ImageGridAmp=ca_f;
cs_c.Time=0:1:dt*num-1;
% cs.ImageGridAmp=ca_f;
% cs.Time=0:1:dt*num-1;
%then use 'import from matlab' option in brainstorm to load new source signals
%import cs_c to cortex map
%import es_c to eeg signals

%save mat
 eeg_r=reshape(eeg_f(chan,:),channel,dt,num);
 eeg_d=permute(eeg_r,[2 3 1]);
%  save(['scpeeg',num2str(snr)],'eeg_r');
%  save(['r_scpeeg',num2str(snr)],'eeg_d');
 aa=eeg_r(:,:,1);



