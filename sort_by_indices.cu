#include <cub/cub.cuh>
#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <thrust/sequence.h>
#include <thrust/device_vector.h>

void SortIndicesByKeys(int* d_keys, int* d_values, int num_items) {

    cudaMalloc(&d_indices, sizeof(keys));   
    // Initialize indices to 0, 1, 2, ..., num_items-1
    thrust::sequence(thrust::device, d_indices, d_indices + num_items);
    void *d_temp_storage = nullptr;
    size_t temp_storage_bytes = 0;

    // Determine temporary storage requirements
    cub::DeviceRadixSort::SortPairs(d_temp_storage, temp_storage_bytes, d_keys, d_indices, num_items);
    cudaMalloc(&d_temp_storage, temp_storage_bytes);
    // Sort indices based on keys
    cub::DeviceRadixSort::SortPairs(d_temp_storage, temp_storage_bytes, d_keys, d_indices, num_items);

    // Retrieve the sorted indices
    std::vector<int> sorted_indices(num_items);
    cudaMemcpy(sorted_indices.data(), d_indices, sizeof(keys), cudaMemcpyDeviceToHost);

    std::cout << "Sorted indices and corresponding key-value pairs:" << std::endl;
    for (int i = 0; i < num_items; ++i) {
        int idx = sorted_indices[i];
        std::cout << "Index: " << idx << ", Key: " << keys[idx] << ", Value: " << values[idx] << std::endl;
    }

    cudaFree(d_indices);
    cudaFree(d_temp_storage);
}

int main() {
    // Example key-value pairs
    int keys[] = {40, 10, 100, 30, 90};
    int values[] = {400, 100, 1000, 300, 900};
    int num_items = sizeof(keys) / sizeof(keys[0]);

    // Device memory allocation
    int *d_keys, *d_values, *d_indices;
    cudaMalloc(&d_keys, sizeof(keys));
    cudaMalloc(&d_values, sizeof(values));
    
    cudaMemcpy(d_keys, keys, sizeof(keys), cudaMemcpyHostToDevice);
    cudaMemcpy(d_values, values, sizeof(values), cudaMemcpyHostToDevice);

    SortIndicesByKeys(d_keys, d_values, num_items);

    cudaFree(d_keys);
    cudaFree(d_values);

    return 0;
}
