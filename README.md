# warp_vs_taichi
compare warp and taichi performance

Usage: 
```
python3 torch_conversion.py -lang taichi
python3 torch_conversion.py -lang warp
```

Findings (cost of conversion between pytorch and taichi/warp):
```
taichi: 0.326s
warp:   0.028s
```
The recorded timing is the total time, i.e., for converting back-and-forth for 1000 times.

- Environment: NVIDIA GeForce RTX 3080.
- Matrix size: 27008 by 3.
