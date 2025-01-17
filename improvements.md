1. Cross attention conflict loss
intuition: based on cross attention map instead of self attention map, so it can both control the target response and the attention conflict. However the paper authors conduct their experiments that replace the self attention map with the cross attention map, and claim the unsatisfactory results are due to in accurate text embedding extracted by CLIP. We think this claim is not convincing.
observation: after some iterations, the cross-attention response loss decreases to a very low value while the self-attention conflict loss remains high. This leads to the optimization process focusing on the self-attention conflict issue and ignoring the cross-attention response one. So we add the cross attention conflict loss (also known as *Attention Segregation Loss* in *AStar* paper) to get a better optimization result.

2. RMSProp optimizer
intuition: the original paper uses Adam optimizer, and it will cause the process to run the whole 5*10 pre-optimization. 
We use RMSProp optimizer to speed up the optimization process, while modifing the tau threshold of cross attention and self conflict loss to get a better result.

3. SD->AE

4. more stable and faster update of the parameters?

5. about final experiments on the report: the fig 7 in the original paper. The dataset can be found in AE