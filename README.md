# procedural-generation-eece5604
Final project for NEU EECE 5604: High Performance Computing

### Current stats:
```cpp
Unoptimized, single threaded CPU Runtime: 5183.5440 ms
Unoptimized P100 GPU Runtime: 44.876 ms
P100 GPU Runtime with Texture Memory: 7.2518 ms
```

### Next step:

Right now, our one kernel is responsible for computing all of the octaves. This is a poor implementation, since the octaves are independent. 

What we should do is compute the octaves in their own kernels. Nvidia GPUs allow for concurrent kernel execution, so this will allow us to:

1. Reduce serial exedcution time of each thread, since it no longer needs to loop through all 12 octaves
2. Enable better **occupancy**, meaning we are better utilizing the available hardware