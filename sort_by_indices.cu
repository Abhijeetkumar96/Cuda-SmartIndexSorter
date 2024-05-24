// nvcc -arch=sm_80 -O3 -std=c++17 -o sort_by_indices sort_by_indices.cu
#include <cub/cub.cuh>
#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <thrust/sequence.h>
#include <thrust/device_vector.h>

void display_device_array(const int* d_arr, const int num_items) {
    std::vector<int> h_arr(num_items);
    cudaMemcpy(h_arr.data(), d_arr, sizeof(int) * num_items, cudaMemcpyDeviceToHost);

    for(auto i : h_arr)
        std::cout << i << " ";
    std::cout << std::endl;
}

void SortIndicesByKeys(const std::vector<int>& keys, const std::vector<int>& values, std::vector<int>& sorted_indices) {

    int num_items = keys.size();
    // Device memory allocation
    int *d_keys, *d_values, *d_keys_sorted;
    cudaMalloc(&d_keys, sizeof(int) * num_items);
    cudaMalloc(&d_keys_sorted, sizeof(int) * num_items);
    cudaMalloc(&d_values, sizeof(int) * num_items);
    
    cudaMemcpy(d_keys, keys.data(), sizeof(int) * num_items, cudaMemcpyHostToDevice);
    cudaMemcpy(d_values, values.data(), sizeof(int) * num_items, cudaMemcpyHostToDevice);

    int* d_indices, *d_indices_sorted;
    cudaMalloc(&d_indices, sizeof(int)* num_items);   
    cudaMalloc(&d_indices_sorted, sizeof(int)* num_items);
    // Initialize indices to 0, 1, 2, ..., num_items-1
    thrust::sequence(thrust::device, d_indices, d_indices + num_items);

    std::cout << "Indices array after thrust::sequence:\n";
    display_device_array(d_indices, num_items);

    void *d_temp_storage = nullptr;
    size_t temp_storage_bytes = 0;

    // Determine temporary storage requirements
    cub::DeviceRadixSort::SortPairs(d_temp_storage, temp_storage_bytes, d_keys, d_keys_sorted, d_indices, d_indices_sorted, num_items);
    cudaMalloc(&d_temp_storage, temp_storage_bytes);
    // Sort indices based on keys
    cub::DeviceRadixSort::SortPairs(d_temp_storage, temp_storage_bytes, d_keys, d_keys_sorted, d_indices, d_indices_sorted, num_items);

    // Retrieve the sorted indices
    sorted_indices.resize(num_items);
    cudaMemcpy(sorted_indices.data(), d_indices_sorted, sizeof(int) * num_items, cudaMemcpyDeviceToHost);

    cudaFree(d_keys);
    cudaFree(d_values);
    cudaFree(d_indices);
    cudaFree(d_temp_storage);
}

int main() {
    // Example key-value pairs
    // std::vector<int> keys = {40, 10, 100, 30, 90};
    // std::vector<int> values = {400, 100, 1000, 300, 900};

    // std::vector<int> keys = {0, 0, 0, 1, 2, 2, 3, 4, 2, 5};
    // std::vector<int> values = {2, 3, 4, 2, 5, 0, 0, 0, 1, 2};

    std::vector<int> keys = {5, 5, 5, 2, 2, 2, 3, 4, 1, 0};
    std::vector<int> values = {2, 3, 4, 1, 0, 5, 5, 5, 2, 2};
    std::vector<int> indices;

    SortIndicesByKeys(keys, values, indices);

    std::cout << "sorted_indices:\n";
    for(auto i : indices)
        std::cout << i << " ";
    std::cout << std::endl;

    std::cout << "Sorted indices and corresponding key-value pairs:" << std::endl;
    for (int i = 0; i < keys.size(); ++i) {
        int idx = indices[i];
        std::cout << "Index: " << idx << ", Key: " << keys[idx] << ", Value: " << values[idx] << std::endl;
    }

    return 0;
}
