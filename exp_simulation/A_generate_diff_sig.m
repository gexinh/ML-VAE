clear all ;clc;
%% notation of simulation of EEG and fMRI
% efs : EEG signals sample frequency
% ffs : fMRI scan frequency
% itv : the interval of event stimulation  
% num : the number of event stimulation 
% fs  : the real world frequency
% ERP_dt : ERP potential duration time
% RT  : scan repeat time of fMRI
% eve : event stimulation module
% ieve : interpolation of event stimulation module  
% ERP_p : ERP potential waveform
% eeg_e : frenquency of event module corresponding to EEG
% eeg_s : source waveform corresponding to EEG
% fmri_e : frenquency of event module corresponding to fMRI
% fmri_b : the BOLD signals of fMRI
%% generate the event stimulate 
% interval=7s
% event num=60
% time=480s
% ERP potential =410ms
% frequency
efs=100;   
ffs=0.5;  
fs=1000;  
% event parameters
num=60;  
itv=7;   
ERP_dt=0.41; 
RT=2;
%% we assume we have [4] sources,which source has different waveform
%%%%%%%%%%%%%%%%%%%% sinusoid sequence
s=4;   % source
f1=20; %sinusoid frequency
eve=cell(1,s);
for i=1:num
    eve{1}(i)=sin(2*pi/f1*i);
end
%%%%%%%%%%%%%%%%%%%% Monotone increasing sequence
for i=1:num
    ap=2/num; %amplitude
    eve{2}(i)=-1+i*ap;
end
%%%%%%%%%%%%%%%%%%%% oscillation decay waveform
for i=1:num
    eve{3}(i)=exp(-0.025*i)*sin(2*pi/f1*i);
end
%%%%%%%%%%%%%%%%%%%% Monotone decreasing sequence
for i=1:num
    ap=2/num; %amplitude
    eve{4}(i)=1-i*ap;
end
%%%%%%%%%%%%%   plot  %%%%%%%%%%%%%%% 
figure(1)
hold on
for i=1:s
    subplot(4,1,i)
    stem(eve{i})
    title(['original event stimulate module corresponding to S',num2str(i)])
end
%% interpolation 
ieve=cell(1,s);
for i=1:s
    if ~isempty(eve{i})
        ieve{i}=upsample(eve{i},itv+1);   %upsample  :itv是7，因此
    else
        ieve{i}=[];
    end
end
%%%%%%%%%%% plot %%%%%%%%%%
 figure(2)
hold on
for i=1:s
    subplot(4,1,i)
    if ~isempty(ieve{i})
        stem(0:size(ieve{i},2)-1,ieve{i})  % 从0开始画图
    else
        plot(ieve{i})
    end
     title(['interpolated event module corresponding to S',num2str(i)])
end
%% event module corresponding to EEG
% we need to upsample because the sample frequency of EEG is 100Hz 
eeg_e=cell(1,s);
for i=1:s
    if ~isempty(ieve{i})
        eeg_e{i}=upsample(ieve{i},efs);   %upsample
    else
        eeg_e{i}=[];
    end
end
%%%%%% plot %%%%%
figure(3)
hold on
for i=1:s
    subplot(4,1,i)
    stem(eeg_e{i})
    title(['upsampling event module of EEG corresponding to S',num2str(i)])
end
%% event module corresponding to fMRI
% we need to downsample because the scan frequency of fMRI is 0.5Hz
% 先统一变成1s的，再降采样到2s
fmri_e=cell(1,4);  %downsample
for i=1:s
    if ~isempty(ieve{i})
       fmri_e{i}=downsample(ieve{i},2);   %downsample
    else
       fmri_e{i}=[];
    end
end
%%%%%% plot %%%%%
figure(4)
hold on
for i=1:s
    subplot(4,1,i)
    if ~isempty(fmri_e{i})
        stem(fmri_e{i})
    else
        stem(fmri_e{i})
    end
    title(['downsampling event module of fMRI corresponding to S',num2str(i)])
end
legend('TR=2s');
%% generate the BOLD signals
% generate HRF function

HRF=cell(1,s);
for i=1:s
    p = [i*2 16 1 1 6 0 32];
    HRF{i}=spm_hrf(RT,p);
end
%%%%%% plot %%%%%
figure(5)
hold on
for i=1:s
    subplot(4,1,i)
    plot(HRF{i})
    title(['HRF function of fMRI corresponding to S',num2str(i)])
end

% execute convolution to obtain the eeg cortex source signals 
fmri_b=cell(1,4);
for i=1:s
    if ~isempty(fmri_e{i})&&~isempty(HRF{i})
       fmri_b{i}=conv(fmri_e{i},HRF{i});
       fmri_b{i}=fmri_b{i}(1:(itv+1)*num*ffs); %去除尾部多余信号
    else
       fmri_b{i}=[];
    end
end
%%%%%% plot %%%%%
figure(6)
hold on
for i=1:s
    subplot(4,1,i)
    plot(fmri_b{i})
    title(['BOLD signals of fMRI corresponding to S',num2str(i)])
end

%% generate the cortex source signals of ERP
% generate ERP potential waveform
% frenquency:100hz
% duration time:410ms
ERP_tmp=ERP_dt*efs;   %ERP potential time points
fe=41;
load ERP_p
% ERP_p=cell(1,s);
% 
% for i=1:ERP_tmp
%    ERP_p{1}(i)=sin(2*pi/fe*(i-1));
%    ERP_p{2}(i)=sin(2*pi*(i-1)/fe-2*pi*3/fe);
%    ERP_p{3}(i)=sin(2*pi*(i-1)/fe-2*pi*5/fe);
%    ERP_p{4}(i)=sin(2*pi*(i-1)/fe-2*pi*7/fe);
% end
%%%%%% plot  %%%%%%%%%
figure(7)
hold on
for i=1:s
    subplot(4,1,i)
    if ~isempty(ERP_p{i})
        plot(0:ERP_tmp-1,ERP_p{i})
    else
        plot(ERP_p{i})
    end
    title(['ERP potential waveform corresponding to S',num2str(i)])
end
% execute convolution to obtain the eeg cortex source signals 
eeg_s=cell(1,s);
for i=1:s
    if ~isempty(eeg_e{i})&&~isempty(ERP_p{i})
        eeg_s{i}=conv(eeg_e{i},ERP_p{i});
        eeg_s{i}=eeg_s{i}(1:(itv+1)*num*efs);  %去除尾部多余信号
    else
        eeg_s{i}=[];
    end
    
end
figure(8)
hold on
for i=1:s
    subplot(4,1,i)
%     if ~isempty(eeg_s{i})
%         plot(0:efs:size(eeg_s{i},2)-100,eeg_s{i})
%     else
        plot(eeg_s{i})
%     end
    title(['EEG source waveform corresponding to S',num2str(i)])
end
save SEEG eeg_s
save fMRI fmri_b
% intercepting the ERP signals
% before we doing this, we should import the EEG source signals to the
% Brainstorm to generate the simulated signal of scouts
