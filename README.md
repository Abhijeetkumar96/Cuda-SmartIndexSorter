# Cuda-SmartIndexSorter
	Welcome to Cuda-SmartIndexSorter, an efficient CUDA-based library designed to sort indices based on corresponding key values without modifying the original key-value pairs. This repository provides high-performance solutions tailored for applications requiring optimized indexing based on key arrays, leveraging the power of NVIDIA's CUDA technology.

## Features
- High Performance: Utilizes CUDA's parallel processing capabilities to ensure high-speed sorting operations.
- Flexibility: Designed to work with any numeric key type that can be sorted using radix sort.
- Minimal Data Movement: Sorts indices rather than actual data, minimizing memory bandwidth usage.
- Easy Integration: Simple API for integration with existing CUDA projects or for new developments.
- Example Code: Includes samples demonstrating how to use the library in various scenarios.

## Requirements
- NVIDIA GPU with CUDA Compute Capability 3.5 or higher
- CUDA Toolkit 10.0 or later
- C++ compiler compatible with the CUDA Toolkit

## Quick Start Guide
Clone the repository to your local machine using:

```bash
git clone https://github.com/Abhijeetkumar96/Cuda-SmartIndexSorter.git
cd Cuda-SmartIndexSorter
```

Compile the sample project:
```bash
nvcc -std=c++17 -O3 -o indexSorter indexSorter.cu
```

## Contributing
Contributions are what make the open-source community such an amazing place to learn, inspire, and create. Any contributions you make are greatly appreciated.

1. Fork the Project
2. Create your Feature Branch (git checkout -b feature/AmazingFeature)
3. Commit your Changes (git commit -m 'Add some AmazingFeature')
4. Push to the Branch (git push origin feature/AmazingFeature)
5. Open a Pull Request

Contact
Abhijeet – cs22s501@iittp.ac.in

Project Link: https://github.com/Abhijeetkumar96/Cuda-SmartIndexSorter

