import warp as wp; 
print('Devices:', [str(d) for d in wp.get_devices()]); 
print('Default:', wp.get_device())