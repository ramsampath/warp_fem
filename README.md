NViDIA Warp Sheet bend metal test

First installation of the virtual environment for python

```
python -m venv warp_env
warp_env\Scripts\activate
```
Then install uv and pip 

```
pip install uv
uv pip install warp-lang numpy usd-core
```

To Test warp installation
```
python .\warp_device_test.py
```

It should display something like
```
Warp 1.8.1 initialized:
   CUDA Toolkit 12.8, Driver 12.6
   Devices:
     "cpu"      : "AMD64 Family 25 Model 1 Stepping 1, AuthenticAMD"
     "cuda:0"   : "NVIDIA GeForce RTX 3060" (12 GiB, sm_86, mempool enabled)
   Kernel cache:
     C:\Users\Ram.CENTROIDLAB\AppData\Local\NVIDIA\warp\Cache\1.8.1
Devices: ['cpu', 'cuda:0']
Default: cuda:0
```

There's results from a current simulation which can be viewer via the USD viewer
```
usdview sheet_metal.usd
```

To run the simulation you can 
```
python .\fem_sheet_metal.py --cell 0.01 --punch_radius 0.04 --punch_speed 0.01 --punch_depth 0.10 --num_steps 200 --stage_path sheet_metal.usd
```

