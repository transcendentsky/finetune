Train
```
CUDA_VISIBLE_DEVICES=0,1 python -m core.ddp --tag debug
```

Test
```
CUDA_VISIBLE_DEVICES=0,1 python -m core.tester --tag debug
```