% 信噪比的定义为
%            信号能量              (纯信号)^2
% SNR=-----------------=--------------------------
%            噪声能量        (带噪信号-纯信号)^2

function snr=SNR_singlech(I,In)
% 计算信噪比函数
% I :original signal
% In:noisy signal(ie. original signal + noise signal)
snr=0;
Ps=sum(sum((I-mean(mean(I))).^2));%signal power
Pn=sum(sum((I-In).^2));           %noise power
snr=10*log10(Ps/Pn);
% 其中I是纯信号，In是带噪信号，snr是信噪比