%验证正演矩阵与实际的关系
load SEEG
itv=8; 
num=60;
efs=100;
erp=cell(1,num);
eeg_f=[];
eeg=[];
dt=41;
for i=1:4
    eeg=cat(1,eeg,eeg_s{i});
end

for i=1:num
    erp{i}=eeg(:,1+(i-1)*itv*efs:dt+(i-1)*itv*efs);
    eeg_f=[eeg_f erp{i}]; %eeg final 
end
%import cortex as cort
atlas=cort.Atlas(9).Scouts;
%% get the ROI vertices
V=cell(1,4);
SNR_check=cell(1,4);
for i=1:4
    V{i}=atlas(i).Vertices;
end
v=size(cort.Vertices,1);
t=size(eeg_f,2);
v_t=zeros(v,t);
for i=1:4
    a=size(V{i}');
    eeg_source=repmat(eeg_f(i,:),a);
     v_t(V{i},:)=eeg_source;
end

eeg_scalp=Gain_matrix*v_t;
