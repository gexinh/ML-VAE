%用来分割数据并存储的脚本，省的每次都需要复用一大段函数
% eeg_o:[chan tmc trail];
function [x_ori x_train x_test ]=split_data(eeg_o,split_portion)
[u j k]=size(eeg_o);
eeg=permute(eeg_o,[3 1 2]);
%保存矩阵
 l=length(eeg);
 l1=split_portion/10*l;
 l2=l1+1;
 s=1/(1-split_portion/10);
 s=int32(s);
x_train=eeg(1:l1,:,:);
x_test=eeg(l2:end,:,:);
x_ori=reshape(permute(x_test,[2 3 1]),[u,j*k/s]);
end