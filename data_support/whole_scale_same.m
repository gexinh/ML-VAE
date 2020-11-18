%将特征缩放到同一个max,min之间
%全部的样本和全部的维度，找出一个最大，最小值。
%% 处理仿真数据
eeg={};
flag=1;
whole_set=[];
portion=8
for i=1:5
 %method 1
    eeg{i}=load(sprintf('eeg/o-run-0%d',i));
    [u j k]=size(eeg{i}.eeg_r); %chan time trial
    a=eeg{i}.eeg_r;
    whole_set=cat(3,whole_set,a);
%     eeg_s=reshape(a,[u*j,k]);
%     m{i}=max(eeg_s);  %max
%     mi{i}=min(eeg_s); %min
%     
end
a=reshape(whole_set,[1,u*j*k*5]);
max_t=max(a);
min_t=min(a);
b=2*(a-min_t)./(max_t-min_t)-1;
eeg_scale=reshape(b,[u,j,k*5]);
eeg_r=reshape(b,[u,j*k*5]);%检查一下放缩的效果
%检测一下逆的效果
load('D:\Project\EEG-fMRI同步\DGMM-test\Gain.mat');
source=pinv(Gain_matrix)*eeg_r(:,1:2460);  %发现源的效果很一般，但是有些
%做到这里，可以检测一下不加噪声的信号的源的复现效果：
% [eeg_scalp,Gain_matrix ,v_t] =gainmatrix_verify();
%保存数据
%eeg_scale：[chan tmc trial]
[X_ori eeg_train eeg_test]=split_data(eeg_scale,portion);
save X X_ori
save max_t max_t
save min_t min_t
save eeg_all_same_scale2 eeg_train eeg_test
save X_all_same_scale2 X_ori