clear all
f1 = 2; f2 = 20; f3 = 50; %simulate frequency
fs=1000; %sample frequency
t=randperm(6000);
t=sort(t);
%数字信号等于（2pi*模拟频率/采样频率）
for i =2000:6000;
Data(1,i) = sin(f1*2*pi*t(i)/fs) ;
%+ 0.4 * cos(f2*2*pi*t(i)/fs);
end
%Data = Data + 0.2 * sin(f3*2*pi*t/fs) + 0.4 * rand(1,6000);
plot(t,Data)