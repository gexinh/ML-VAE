clear all
%% event related module
 %time interval:5s
 %event amplitude:differnt source corresponding to different sinusoid
 %event count:120
intv=5;
num=120;
% use different sample frequency corresponding to different modality
ff=0.5;  %fMRI frequency =0.5hz
fs=1000;  %sample frequency=1000hz ,thus time point can be accurate to 'ms' level
tmp=num*intv*fs; %time point
event=[];
for i=1:tmp+1 
    if mod(i,fs*intv)==0
       event(i)=1;
    else
       event(i)=0;
    end
end
figure(1)
stem(0:0.001:intv*num,event); %画图时候采样点从0开始计算
title({'the stimulate module';['the number of event is ',num2str(num)]})
%% the source of EEG signals
%ERP duration time:400ms
%ERP potential waveform:differnt sources corresponding to different waveform
%sampling frequency:200HZ
%how to generate:ERP potential signals conv stimulate signal ,and
%intercepting the 

%% fMRI signals
%HRF duration point:17 
%different source corresponding to different shifted HRF functions 
RT=1; % scan time(TR) =2s 1s 0.5s
len=32;% HRF duration time
p = [6 16 1 1 6 0 len];
HRF=spm_hrf(RT,p);%the waveform and amplitude of HRF depents on the TR.
HRF2=spm_hrf(1,p);
figure(2)
subplot(211)
plot(0:RT:32,HRF) %this is the real waveform of HRF because of TR=2s,
title({'the HRF function',
       ['sample frequency is ',num2str(RT)]})
subplot(212)
plot(0:1:32,HRF2)
title({'the HRF function',
        ['sample frequency is 1']})
%however the HRF only have [len*fs] sample point,thus we must downsample 
%the event related module to 0.5Hz before we convolution HRF.
%in order to record sturation BOLD signals, we must add some timecourse
%in the end of the event related module, we will choose 30s in here
%st=30; %saturation time
%tmpf=intv*num+st;
tmpf=intv*num;
ffs=1/RT;            %sample frequency of fMRI 
ftc=tmpf*ffs;        %fMRI time course
fe=[];              %the event stimulate in the sample frequency of fMRI
for j=1:ftc
    if mod(j,intv*ffs)==0
        fe(j)=1;
    else
        fe(j)=0;
    end
end
%% caculate the BOLD signals
fmri=conv(fe,HRF);    %convolution
fmri=fmri(1:end-1);   %

%% plot
figure(3)
subplot(211)
%plot(0:RT:600,fe);
stem(0:RT:size(fe,2)-1,fe);
subplot(212)
plot(fmri);

% we find the BOLD signals will be a Periodic Oscillation curve with the 
% equivalent stimulate signals. Thus, we need to simulate some diffenrent signals
% we will use another script to generate those signals. 
% the detail please refer to : [diff_signals.m]