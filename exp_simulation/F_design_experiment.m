%% train and test is :9:1
s=[0 10 15 25];
for i=1:4;
    snr=s(i);
    load(['scpeeg',num2str(snr)])
    a=size(eeg_r,3);
    t=a/10;
    eeg_train=eeg_r(:,:,1:a-t);
    eeg_test=eeg_r(:,:,a-t+1:end);
    save(['eeg',num2str(snr)],'eeg_train','eeg_test')
end


