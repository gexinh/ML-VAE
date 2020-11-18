%将数据拼接成DGMM需要的数据，注意里面的fMRI部分并不好用，改成了SVD分解
eeg=cell(1,1,5);
fmri=cell(1,5);
for i=1:5
 %method 1
    eeg{1,1,i}=load(sprintf('eeg/o-run-0%d',i));
 %method 2
    eeg{i}=load(['eeg/o-run-0',num2str(i)]);
    eeg{1,1,i}=eeg{1,1,i}.eeg_r;
    fmri{1,i}=load(['./fun/run-0',num2str(i)]);
    fmri{1,i}=fmri{1,i}.fmri_bold; 
end
eeg_c=cell2mat(eeg);
fmri_c=cell2mat(fmri);
%因为fMRI时间上分辨率较低，因此DGMM里一个trial只有一幅切片
%测试集与样本集应该相互独立
eeg_c=permute(eeg_c,[3 1 2]);
fmri_c=permute(fmri_c,[2 1]);
eeg_train=eeg_c(1:240,:,:);
eeg_test=eeg_c(241:end,:,:);
fmri_train=fmri_c(1:240*4,:);
fmri_test=fmri_c(240*4+1:end,:);

save eeg eeg_train eeg_test
save fmri fmri_train fmri_test
