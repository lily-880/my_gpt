# INT8 对照样例（prompt='ROMEO:', temp=0.8, top_k=50, seed=0)

同一 seed 各抽。量化会改变 logits，文本不必相同。不要 greedy。

## fp32 sample 1

```
ROMEO: ” ” was not the 
mang'd. No, not I, but that now did speak 
 That I did find a man as thou art, 
 So should I
```

## fp32 sample 2

```
ROMEO: 
No more than this is true. But when it comes, 
For this is done; therefore ’tis too ill 
Of that we will not speak; and so I know
```

## int8 sample 1

```
ROMEO: ” ” was not the 
mantwere to be the villain. 
 
LADY CAPTAIN. 
O, why should it be. 
 

```

## int8 sample 2

```
ROMEO: 
No more thou best nor name thee, to love thee. 
 
’Tis not to die, than honour be true servant. 
 
NURSE.
```
