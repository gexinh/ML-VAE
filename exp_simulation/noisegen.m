function [Y,NOISE] = noisegen(X,SNR)
% noisegen add white Gaussian noise to a signal.
% X：N*T  n-sample t-timecourse
% [Y, NOISE] = NOISEGEN(X,SNR) adds white Gaussian NOISE to X.  The SNR is in dB.
NOISE=randn(size(X)); 
% [num tmc]=size(X);
% for i=1:num
%    noise(i,:)=randn(1,tmc)
% end
NOISE=NOISE-mean(NOISE,2);                           %去均值/中心化
signal_power = var(X,0,2); %等价：1/(size(bold,2)-1)*sum(X.*X,2);
noise_variance = signal_power ./ ( 10^(SNR/10) );   %noise power=noise variance
NOISE=sqrt(noise_variance)./std(NOISE,0,2).*NOISE;       %noise放大标准差的倍数
Y=X+NOISE;                 