efs=100;   
ffs=0.5;  
fs=1000;  
% event parameters
num=60;  
itv=7;   
ERP_dt=0.41; 
s=4;

%% generate the cortex source signals of ERP
% generate ERP potential waveform
% frenquency:100hz
% duration time:410ms
ERP_tmp=ERP_dt*efs;   %ERP potential time points
ERP_p=cell(1,s);
%ERP 1
x=[0 0.8 0.7 0 -0.4 -0.2 0 0 0];
ERP_p{1}=interp(x,4);
ERP_p{1}=[zeros(1,5) ERP_p{1}];

%ERP 2
x=[0 0 0 0 0.8 0.8 0 -0.5 0 0 ];
ERP_p{2}=interp(x,4);
n=randn(1,41)*0.001;

ERP_p{2}=[zeros(1,7) ERP_p{2}(1,1:33) 0];
ERP_p{2}=ERP_p{2}+n;

%ERP 3
x=[0 0 0 0.1 0.8 0.8 0.1 0 0 0 ];
ERP_p{3}=interp(x,4);
ERP_p{3}=ERP_p{3}(1,7:37);
ERP_p{3}=[zeros(1,3) ERP_p{3} zeros(1,7)]+n;

%ERP 4
p = [6 12 1 1 6 0 40];
ERP_p{4}=spm_hrf(1,p);
ERP_p{4}=ERP_p{4}*4;
%再以21为中心做翻转
for i=1:20
    ERP_p{4}(9)=1;
    ERP_p{4}(21+i)=ERP_p{4}(21-i);
end
ERP_p{4}=ERP_p{4}';
% save 
save ERP_p ERP_p
%%%%%% plot  %%%%%%%%%
% figure(7)
% hold on
% for i=1:s
%     subplot(4,1,i)
%     if ~isempty(ERP_p{i})
%         plot(0:ERP_tmp-1,ERP_p{i})
%     else
%         plot(ERP_p{i})
%     end
%     title(['ERP potential waveform corresponding to S',num2str(i)])
% end