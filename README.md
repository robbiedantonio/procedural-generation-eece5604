# procedural-generation-eece5604
Final project for NEU EECE 5604: High Performance Computing

### Current stats:
```c
Unoptimized, single threaded CPU Runtime: 5183.5440 ms
P100, one kernel per octave streamed, accumulator kernel: 54.76 ms
P100, one kernel computes and sums all octaves: 34.48 ms
P100, three-dimensional kernel: 56.99 ms 
P100, three-dimension kernel + texture memory: 39.99 ms
P100 GPU Runtime with Texture Memory: 4.62 ms
H100 GPU Runtime with texture memory: 2.87 ms
```

### Next step:

Right now, our one kernel is responsible for computing all of the octaves. This is a poor implementation, since the octaves are independent. 

What we should do is compute the octaves in their own kernels. Nvidia GPUs allow for concurrent kernel execution, so this will allow us to:

1. Reduce serial exedcution time of each thread, since it no longer needs to loop through all 12 octaves
2. Enable better **occupancy**, meaning we are better utilizing the available hardware
3. As it turns out, this gets completely offset by this implementations memory requirement