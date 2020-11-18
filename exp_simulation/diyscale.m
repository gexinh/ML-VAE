%æÿ’Û∑≈Àı
function [new_s]=diyscale(sig,mask,new_min,new_max)
 non0signals=sig(mask,:);
 [l w]=size(non0signals);
 non0sig_r=reshape(non0signals,l*w,1);
 old_max=max(non0sig_r);
 old_min=min(non0sig_r);
 %scale
 diff=old_max-old_min;
 diff2=new_max-new_min;
 scale=diff2/diff;
 new=(non0sig_r-old_min).*scale;
 new_r=reshape(new,[l w]);
 sig(mask,:)=new_r;
 
 %output
 new_s=sig; 