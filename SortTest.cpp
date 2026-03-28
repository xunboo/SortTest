// SortTest.cpp : This file contains the 'main' function. Program execution begins and ends there.
//

#include <algorithm>
#include <chrono>
#if defined(__has_include)
#if __has_include(<execution>)
#include <execution>
#define HAS_STD_EXECUTION 1
#endif
#endif
#ifndef HAS_STD_EXECUTION
#define HAS_STD_EXECUTION 0
#endif
#if defined(__has_include)
#if __has_include(<thrust/device_vector.h>) && __has_include(<thrust/sort.h>)
#include <thrust/device_vector.h>
#include <thrust/sort.h>
#define HAS_THRUST_CUDA 1
#endif
#endif
#ifndef HAS_THRUST_CUDA
#define HAS_THRUST_CUDA 0
#endif
#if defined(__has_include)
#if __has_include(<CL/cl.h>)
#include <CL/cl.h>
#define HAS_OPENCL 1
#elif __has_include(<OpenCL/opencl.h>)
#include <OpenCL/opencl.h>
#define HAS_OPENCL 1
#endif
#endif
#ifndef HAS_OPENCL
#define HAS_OPENCL 0
#endif
#if defined(_OPENMP)
#include <omp.h>
#define HAS_OPENMP 1
#else
#define HAS_OPENMP 0
#endif
#include <cstdlib>
#include <condition_variable>
#include <cstring>
#include <iostream>
#include <limits>
#include <mutex>
#include <random>
#include <string>
#include <thread>
#include <vector>

long long claude_sort(std::vector<int>& values);

long long stl_sort(std::vector<int>& values)
{
    const auto start = std::chrono::high_resolution_clock::now();
    std::sort(values.begin(), values.end());
    const auto end = std::chrono::high_resolution_clock::now();

    return std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
}

long long multithread_sort(std::vector<int>& values)
{
    const auto start = std::chrono::high_resolution_clock::now();

    const std::size_t n = values.size();
    if (n <= 1)
    {
        return 0;
    }

    unsigned int threadCount = std::thread::hardware_concurrency();
    if (threadCount == 0)
    {
        threadCount = 4;
    }
    if (threadCount > n)
    {
        threadCount = static_cast<unsigned int>(n);
    }

    std::vector<std::size_t> boundaries;
    boundaries.reserve(static_cast<std::size_t>(threadCount) + 1);
    boundaries.push_back(0);

    const std::size_t baseChunk = n / threadCount;
    const std::size_t remainder = n % threadCount;

    std::size_t offset = 0;
    for (unsigned int i = 0; i < threadCount; ++i)
    {
        const std::size_t chunkSize = baseChunk + (i < remainder ? 1 : 0);
        offset += chunkSize;
        boundaries.push_back(offset);
    }

    std::vector<std::thread> workers;
    workers.reserve(threadCount);

    for (unsigned int i = 0; i < threadCount; ++i)
    {
        const std::size_t begin = boundaries[i];
        const std::size_t end = boundaries[i + 1];
        workers.emplace_back([&values, begin, end]() {
            std::sort(values.begin() + static_cast<std::ptrdiff_t>(begin),
                      values.begin() + static_cast<std::ptrdiff_t>(end));
        });
    }

    for (auto& worker : workers)
    {
        worker.join();
    }

    std::size_t currentBlockCount = threadCount;
    while (currentBlockCount > 1)
    {
        std::size_t writeIndex = 0;
        for (std::size_t i = 0; i + 1 < currentBlockCount; i += 2)
        {
            const std::size_t leftBegin = boundaries[i];
            const std::size_t middle = boundaries[i + 1];
            const std::size_t rightEnd = boundaries[i + 2];

            std::inplace_merge(values.begin() + static_cast<std::ptrdiff_t>(leftBegin),
                               values.begin() + static_cast<std::ptrdiff_t>(middle),
                               values.begin() + static_cast<std::ptrdiff_t>(rightEnd));

            boundaries[writeIndex] = leftBegin;
            ++writeIndex;
        }

        if (currentBlockCount % 2 == 1)
        {
            boundaries[writeIndex] = boundaries[currentBlockCount - 1];
            ++writeIndex;
        }

        boundaries[writeIndex] = boundaries[currentBlockCount];
        currentBlockCount = writeIndex;
    }

    const auto end = std::chrono::high_resolution_clock::now();
    return std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
}

long long parallel_stl_sort(std::vector<int>& values)
{
    const auto start = std::chrono::high_resolution_clock::now();

#if HAS_STD_EXECUTION && ((defined(_MSVC_LANG) && _MSVC_LANG >= 201703L) || __cplusplus >= 201703L)
    std::sort(std::execution::par, values.begin(), values.end());
#else
    std::sort(values.begin(), values.end());
#endif

    const auto end = std::chrono::high_resolution_clock::now();
    return std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
}

#if HAS_OPENCL
bool opencl_sort(std::vector<int>& values)
{
    if (values.empty())
    {
        return true;
    }

    cl_int err = CL_SUCCESS;

    cl_uint platformCount = 0;
    err = clGetPlatformIDs(0, nullptr, &platformCount);
    if (err != CL_SUCCESS || platformCount == 0)
    {
        return false;
    }

    std::vector<cl_platform_id> platforms(platformCount);
    err = clGetPlatformIDs(platformCount, platforms.data(), nullptr);
    if (err != CL_SUCCESS)
    {
        return false;
    }

    cl_device_id device = nullptr;
    for (std::size_t i = 0; i < platforms.size(); ++i)
    {
        cl_uint deviceCount = 0;
        err = clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_GPU, 0, nullptr, &deviceCount);
        if (err == CL_SUCCESS && deviceCount > 0)
        {
            std::vector<cl_device_id> devices(deviceCount);
            err = clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_GPU, deviceCount, devices.data(), nullptr);
            if (err == CL_SUCCESS)
            {
                device = devices[0];
                break;
            }
        }
    }

    if (device == nullptr)
    {
        return false;
    }

    cl_context context = clCreateContext(nullptr, 1, &device, nullptr, nullptr, &err);
    if (err != CL_SUCCESS || context == nullptr)
    {
        return false;
    }

    cl_command_queue queue = clCreateCommandQueueWithProperties(context, device, nullptr, &err);
    if (err != CL_SUCCESS || queue == nullptr)
    {
        clReleaseContext(context);
        return false;
    }

    // GPU LSD radix sort — 4 passes of 8-bit digit, O(4n) vs O(n log²n) for bitonic.
    //
    // radix_flip   : int -> uint via sign-bit XOR so unsigned order = signed order
    // radix_count  : each work group counts its chunk into __local histogram (local
    //                atomics are fast scratchpad ops), then writes to global hist buffer
    // radix_scatter: loads per-group offsets into __local, each thread atomically claims
    //                its output slot — 512 work groups scatter in parallel, zero contention
    //                between groups because their output regions are disjoint by construction
    // radix_unflip : uint -> int sign-bit XOR restore
    //
    // The prefix sum between count and scatter runs on the CPU (512 groups * 256 buckets
    // = 131 072 uints) — negligible time, avoids a separate GPU scan kernel.
    const char* kernelSource = R"kernel(
#pragma OPENCL EXTENSION cl_khr_global_int32_base_atomics : enable
#pragma OPENCL EXTENSION cl_khr_local_int32_base_atomics  : enable

__kernel void radix_flip(__global const int* src, __global uint* dst, const uint n)
{
    const uint gid = get_global_id(0);
    if (gid < n)
        dst[gid] = (uint)src[gid] ^ 0x80000000u;
}

// Each work group counts its contiguous chunk of `in` into a 256-entry local histogram,
// then writes that histogram to global memory at hist[gid * 256 + b].
__kernel void radix_count(__global const uint* in,
                          __global uint*       hist,
                          const uint           n,
                          const uint           shift,
                          const uint           chunkSize)
{
    __local uint lhist[256];
    const uint lid   = get_local_id(0);
    const uint gid   = get_group_id(0);
    const uint lsize = get_local_size(0);

    for (uint b = lid; b < 256u; b += lsize)
        lhist[b] = 0u;
    barrier(CLK_LOCAL_MEM_FENCE);

    const uint start = gid * chunkSize;
    const uint end   = (start + chunkSize < n) ? start + chunkSize : n;

    for (uint i = start + lid; i < end; i += lsize)
        atomic_add(&lhist[(in[i] >> shift) & 0xFFu], 1u);
    barrier(CLK_LOCAL_MEM_FENCE);

    for (uint b = lid; b < 256u; b += lsize)
        hist[gid * 256u + b] = lhist[b];
}

// Each work group loads its per-bucket output offsets into __local memory, then each
// thread atomically claims a slot in that bucket and writes its element there.
// Different work groups write to disjoint output regions — no inter-group contention.
__kernel void radix_scatter(__global const uint* in,
                            __global uint*       out,
                            __global const uint* offsets,
                            const uint           n,
                            const uint           shift,
                            const uint           chunkSize)
{
    __local uint lpos[256];
    const uint lid   = get_local_id(0);
    const uint gid   = get_group_id(0);
    const uint lsize = get_local_size(0);

    for (uint b = lid; b < 256u; b += lsize)
        lpos[b] = offsets[gid * 256u + b];
    barrier(CLK_LOCAL_MEM_FENCE);

    const uint start = gid * chunkSize;
    const uint end   = (start + chunkSize < n) ? start + chunkSize : n;

    for (uint i = start + lid; i < end; i += lsize)
    {
        const uint v   = in[i];
        const uint b   = (v >> shift) & 0xFFu;
        const uint pos = atomic_add(&lpos[b], 1u);
        out[pos] = v;
    }
}

__kernel void radix_unflip(__global const uint* src, __global int* dst, const uint n)
{
    const uint gid = get_global_id(0);
    if (gid < n)
        dst[gid] = (int)(src[gid] ^ 0x80000000u);
}
)kernel";

    cl_program program = clCreateProgramWithSource(context, 1, &kernelSource, nullptr, &err);
    if (err != CL_SUCCESS || program == nullptr)
    {
        clReleaseCommandQueue(queue);
        clReleaseContext(context);
        return false;
    }

    err = clBuildProgram(program, 1, &device, nullptr, nullptr, nullptr);
    if (err != CL_SUCCESS)
    {
        clReleaseProgram(program);
        clReleaseCommandQueue(queue);
        clReleaseContext(context);
        return false;
    }

    cl_kernel kFlip    = clCreateKernel(program, "radix_flip",    &err);
    if (err != CL_SUCCESS || kFlip == nullptr)
    {
        clReleaseProgram(program); clReleaseCommandQueue(queue); clReleaseContext(context);
        return false;
    }
    cl_kernel kCount   = clCreateKernel(program, "radix_count",   &err);
    if (err != CL_SUCCESS || kCount == nullptr)
    {
        clReleaseKernel(kFlip);
        clReleaseProgram(program); clReleaseCommandQueue(queue); clReleaseContext(context);
        return false;
    }
    cl_kernel kScatter = clCreateKernel(program, "radix_scatter", &err);
    if (err != CL_SUCCESS || kScatter == nullptr)
    {
        clReleaseKernel(kCount); clReleaseKernel(kFlip);
        clReleaseProgram(program); clReleaseCommandQueue(queue); clReleaseContext(context);
        return false;
    }
    cl_kernel kUnflip  = clCreateKernel(program, "radix_unflip",  &err);
    if (err != CL_SUCCESS || kUnflip == nullptr)
    {
        clReleaseKernel(kScatter); clReleaseKernel(kCount); clReleaseKernel(kFlip);
        clReleaseProgram(program); clReleaseCommandQueue(queue); clReleaseContext(context);
        return false;
    }

    const cl_uint n          = static_cast<cl_uint>(values.size());
    const cl_uint WG_SIZE    = 256u;
    const cl_uint NUM_GROUPS = 512u;
    const cl_uint CHUNK_SIZE = (n + NUM_GROUPS - 1u) / NUM_GROUPS;

    // bufA / bufB ping-pong during sort; both hold n * 4 bytes (int and uint are same size)
    cl_mem bufA = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                                 n * sizeof(cl_int), values.data(), &err);
    if (err != CL_SUCCESS || bufA == nullptr)
    {
        clReleaseKernel(kUnflip); clReleaseKernel(kScatter);
        clReleaseKernel(kCount);  clReleaseKernel(kFlip);
        clReleaseProgram(program); clReleaseCommandQueue(queue); clReleaseContext(context);
        return false;
    }
    cl_mem bufB = clCreateBuffer(context, CL_MEM_READ_WRITE,
                                 n * sizeof(cl_uint), nullptr, &err);
    if (err != CL_SUCCESS || bufB == nullptr)
    {
        clReleaseMemObject(bufA);
        clReleaseKernel(kUnflip); clReleaseKernel(kScatter);
        clReleaseKernel(kCount);  clReleaseKernel(kFlip);
        clReleaseProgram(program); clReleaseCommandQueue(queue); clReleaseContext(context);
        return false;
    }
    // hist[gid * 256 + b] = count of digit b in work group gid's chunk
    cl_mem bufHist = clCreateBuffer(context, CL_MEM_READ_WRITE,
                                    NUM_GROUPS * 256u * sizeof(cl_uint), nullptr, &err);
    if (err != CL_SUCCESS || bufHist == nullptr)
    {
        clReleaseMemObject(bufB); clReleaseMemObject(bufA);
        clReleaseKernel(kUnflip); clReleaseKernel(kScatter);
        clReleaseKernel(kCount);  clReleaseKernel(kFlip);
        clReleaseProgram(program); clReleaseCommandQueue(queue); clReleaseContext(context);
        return false;
    }
    // offsets[gid * 256 + b] = first global output index for work group gid's bucket b
    cl_mem bufOffs = clCreateBuffer(context, CL_MEM_READ_WRITE,
                                    NUM_GROUPS * 256u * sizeof(cl_uint), nullptr, &err);
    if (err != CL_SUCCESS || bufOffs == nullptr)
    {
        clReleaseMemObject(bufHist); clReleaseMemObject(bufB); clReleaseMemObject(bufA);
        clReleaseKernel(kUnflip); clReleaseKernel(kScatter);
        clReleaseKernel(kCount);  clReleaseKernel(kFlip);
        clReleaseProgram(program); clReleaseCommandQueue(queue); clReleaseContext(context);
        return false;
    }

    const std::size_t flipGWS = ((static_cast<std::size_t>(n) + WG_SIZE - 1u) / WG_SIZE) * WG_SIZE;
    const std::size_t flipLWS = WG_SIZE;
    const std::size_t sortGWS = static_cast<std::size_t>(NUM_GROUPS) * WG_SIZE;
    const std::size_t sortLWS = WG_SIZE;

    // --- Flip: int -> uint with sign-bit XOR (bufA[int] -> bufB[uint]), then swap ---
    err  = clSetKernelArg(kFlip, 0, sizeof(cl_mem),  &bufA);
    err |= clSetKernelArg(kFlip, 1, sizeof(cl_mem),  &bufB);
    err |= clSetKernelArg(kFlip, 2, sizeof(cl_uint), &n);
    err |= clEnqueueNDRangeKernel(queue, kFlip, 1, nullptr, &flipGWS, &flipLWS, 0, nullptr, nullptr);
    if (err != CL_SUCCESS)
    {
        clReleaseMemObject(bufOffs); clReleaseMemObject(bufHist);
        clReleaseMemObject(bufB);    clReleaseMemObject(bufA);
        clReleaseKernel(kUnflip); clReleaseKernel(kScatter);
        clReleaseKernel(kCount);  clReleaseKernel(kFlip);
        clReleaseProgram(program); clReleaseCommandQueue(queue); clReleaseContext(context);
        return false;
    }
    std::swap(bufA, bufB);   // bufA now holds flipped uint data

    // CPU-side prefix sum buffers (512 * 256 = 131 072 uints, trivially small)
    std::vector<cl_uint> hist(NUM_GROUPS * 256u);
    std::vector<cl_uint> offs(NUM_GROUPS * 256u);

    bool success = true;

    // --- 4 radix passes ---
    for (cl_uint pass = 0u; pass < 4u && success; ++pass)
    {
        const cl_uint shift = pass * 8u;

        // Count: each work group counts its chunk into local hist, writes to bufHist
        err  = clSetKernelArg(kCount, 0, sizeof(cl_mem),  &bufA);
        err |= clSetKernelArg(kCount, 1, sizeof(cl_mem),  &bufHist);
        err |= clSetKernelArg(kCount, 2, sizeof(cl_uint), &n);
        err |= clSetKernelArg(kCount, 3, sizeof(cl_uint), &shift);
        err |= clSetKernelArg(kCount, 4, sizeof(cl_uint), &CHUNK_SIZE);
        err |= clEnqueueNDRangeKernel(queue, kCount, 1, nullptr, &sortGWS, &sortLWS, 0, nullptr, nullptr);
        // Blocking read so CPU prefix sum sees finished histogram
        err |= clEnqueueReadBuffer(queue, bufHist, CL_TRUE, 0,
                                   hist.size() * sizeof(cl_uint), hist.data(), 0, nullptr, nullptr);
        if (err != CL_SUCCESS) { success = false; break; }

        // CPU prefix sum: compute per-group scatter offsets (O(256 * NUM_GROUPS) — trivial)
        cl_uint running = 0u;
        for (cl_uint b = 0u; b < 256u; ++b)
        {
            // Sum this bucket across all groups to find its global start
            cl_uint bucketTotal = 0u;
            for (cl_uint g = 0u; g < NUM_GROUPS; ++g)
                bucketTotal += hist[g * 256u + b];
            const cl_uint bucketStart = running;
            running += bucketTotal;
            // Assign per-group offsets within this bucket
            cl_uint pos = bucketStart;
            for (cl_uint g = 0u; g < NUM_GROUPS; ++g)
            {
                offs[g * 256u + b] = pos;
                pos += hist[g * 256u + b];
            }
        }

        // Scatter: each work group uses its local offsets to scatter its chunk into bufB
        err  = clEnqueueWriteBuffer(queue, bufOffs, CL_FALSE, 0,
                                    offs.size() * sizeof(cl_uint), offs.data(), 0, nullptr, nullptr);
        err |= clSetKernelArg(kScatter, 0, sizeof(cl_mem),  &bufA);
        err |= clSetKernelArg(kScatter, 1, sizeof(cl_mem),  &bufB);
        err |= clSetKernelArg(kScatter, 2, sizeof(cl_mem),  &bufOffs);
        err |= clSetKernelArg(kScatter, 3, sizeof(cl_uint), &n);
        err |= clSetKernelArg(kScatter, 4, sizeof(cl_uint), &shift);
        err |= clSetKernelArg(kScatter, 5, sizeof(cl_uint), &CHUNK_SIZE);
        err |= clEnqueueNDRangeKernel(queue, kScatter, 1, nullptr, &sortGWS, &sortLWS, 0, nullptr, nullptr);
        if (err != CL_SUCCESS) { success = false; break; }

        std::swap(bufA, bufB);   // sorted output for this pass becomes input for next
    }

    // --- Unflip: uint -> int (bufA -> bufB), then read bufB back to host ---
    if (success)
    {
        err  = clSetKernelArg(kUnflip, 0, sizeof(cl_mem),  &bufA);
        err |= clSetKernelArg(kUnflip, 1, sizeof(cl_mem),  &bufB);
        err |= clSetKernelArg(kUnflip, 2, sizeof(cl_uint), &n);
        err |= clEnqueueNDRangeKernel(queue, kUnflip, 1, nullptr, &flipGWS, &flipLWS, 0, nullptr, nullptr);
        err |= clEnqueueReadBuffer(queue, bufB, CL_TRUE, 0,
                                   n * sizeof(cl_int), values.data(), 0, nullptr, nullptr);
        if (err != CL_SUCCESS)
            success = false;
    }

    clReleaseMemObject(bufOffs);
    clReleaseMemObject(bufHist);
    clReleaseMemObject(bufB);
    clReleaseMemObject(bufA);
    clReleaseKernel(kUnflip);
    clReleaseKernel(kScatter);
    clReleaseKernel(kCount);
    clReleaseKernel(kFlip);
    clReleaseProgram(program);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);

    return success;
}
#endif

long long gpu_sort(std::vector<int>& values)
{
    const auto start = std::chrono::high_resolution_clock::now();

#if HAS_THRUST_CUDA
    thrust::device_vector<int> d_values(values.begin(), values.end());
    thrust::sort(d_values.begin(), d_values.end());
    thrust::copy(d_values.begin(), d_values.end(), values.begin());
#elif HAS_OPENCL
    if (!opencl_sort(values))
    {
        std::cerr << "Error: GPU sort is not supported on this system (OpenCL GPU unavailable). Falling back to CPU sort.\n";
        std::sort(values.begin(), values.end());
    }
#else
    std::cerr << "Error: GPU sort is not supported in this build (no CUDA/OpenCL backend). Falling back to CPU sort.\n";
    std::sort(values.begin(), values.end());
#endif

    const auto end = std::chrono::high_resolution_clock::now();
    return std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
}

long long codex_sort(std::vector<int>& values, std::string& backend)
{
    const auto start = std::chrono::high_resolution_clock::now();

    const std::size_t n = values.size();
    if (n <= 1)
    {
        backend = "parallel radix 11/11/10";
        const auto end = std::chrono::high_resolution_clock::now();
        return std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    }

    if (n > static_cast<std::size_t>(std::numeric_limits<unsigned int>::max()))
    {
        backend = "std::sort fallback";
        std::sort(values.begin(), values.end());
        const auto end = std::chrono::high_resolution_clock::now();
        return std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    }

    unsigned int threadCount = std::thread::hardware_concurrency();
    if (threadCount == 0)
    {
        threadCount = 4;
    }
    if (threadCount > n)
    {
        threadCount = static_cast<unsigned int>(n);
    }

    constexpr unsigned int MAX_BUCKETS = 2048; // max(2^11, 2^10)
    constexpr unsigned int PASSES = 3;
    constexpr unsigned int PASS_BITS[PASSES] = { 11, 11, 10 };
    constexpr unsigned int PASS_SHIFT[PASSES] = { 0, 11, 22 };

    std::vector<std::size_t> chunkStart(threadCount + 1);
    {
        const std::size_t base = n / threadCount;
        const std::size_t rem = n % threadCount;
        std::size_t pos = 0;
        for (unsigned int t = 0; t < threadCount; ++t)
        {
            chunkStart[t] = pos;
            pos += base + (t < rem ? 1u : 0u);
        }
        chunkStart[threadCount] = n;
    }

    std::vector<unsigned int> in(n);
    std::vector<unsigned int> out(n);

    {
        std::vector<std::thread> workers;
        workers.reserve(threadCount);
        for (unsigned int t = 0; t < threadCount; ++t)
        {
            workers.emplace_back([&, t]() {
                const std::size_t s = chunkStart[t];
                const std::size_t e = chunkStart[t + 1];
                for (std::size_t i = s; i < e; ++i)
                {
                    in[i] = static_cast<unsigned int>(values[i]) ^ 0x80000000u;
                }
            });
        }
        for (auto& w : workers)
        {
            w.join();
        }
    }

    std::vector<std::size_t> hist(static_cast<std::size_t>(threadCount) * MAX_BUCKETS);
    std::vector<std::size_t> baseOffset(MAX_BUCKETS);
    std::vector<std::size_t> posFlat(static_cast<std::size_t>(threadCount) * MAX_BUCKETS);

    std::vector<std::thread> workers;
    workers.reserve(threadCount);

    for (unsigned int pass = 0; pass < PASSES; ++pass)
    {
        const unsigned int bits = PASS_BITS[pass];
        const unsigned int shift = PASS_SHIFT[pass];
        const unsigned int bucketCount = 1u << bits;
        const unsigned int mask = bucketCount - 1u;

        for (unsigned int t = 0; t < threadCount; ++t)
        {
            std::size_t* localHist = hist.data() + static_cast<std::size_t>(t) * MAX_BUCKETS;
            std::fill(localHist, localHist + bucketCount, 0);
        }

        workers.clear();
        for (unsigned int t = 0; t < threadCount; ++t)
        {
            workers.emplace_back([&, t]() {
                std::size_t* localHist = hist.data() + static_cast<std::size_t>(t) * MAX_BUCKETS;
                const std::size_t s = chunkStart[t];
                const std::size_t e = chunkStart[t + 1];
                for (std::size_t i = s; i < e; ++i)
                {
                    const unsigned int b = (in[i] >> shift) & mask;
                    ++localHist[b];
                }
            });
        }
        for (auto& w : workers)
        {
            w.join();
        }

        std::size_t running = 0;
        for (unsigned int b = 0; b < bucketCount; ++b)
        {
            std::size_t sum = 0;
            for (unsigned int t = 0; t < threadCount; ++t)
            {
                const std::size_t idx = static_cast<std::size_t>(t) * MAX_BUCKETS + b;
                const std::size_t c = hist[idx];
                hist[idx] = sum;
                sum += c;
            }
            baseOffset[b] = running;
            running += sum;
        }

        workers.clear();
        for (unsigned int t = 0; t < threadCount; ++t)
        {
            workers.emplace_back([&, t]() {
                std::size_t* pos = posFlat.data() + static_cast<std::size_t>(t) * MAX_BUCKETS;
                std::size_t* threadPrefix = hist.data() + static_cast<std::size_t>(t) * MAX_BUCKETS;

                for (unsigned int b = 0; b < bucketCount; ++b)
                {
                    pos[b] = baseOffset[b] + threadPrefix[b];
                }

                const std::size_t s = chunkStart[t];
                const std::size_t e = chunkStart[t + 1];
                for (std::size_t i = s; i < e; ++i)
                {
                    const unsigned int v = in[i];
                    const unsigned int b = (v >> shift) & mask;
                    out[pos[b]++] = v;
                }
            });
        }
        for (auto& w : workers)
        {
            w.join();
        }

        in.swap(out);
    }

    {
        std::vector<std::thread> finalWorkers;
        finalWorkers.reserve(threadCount);
        for (unsigned int t = 0; t < threadCount; ++t)
        {
            finalWorkers.emplace_back([&, t]() {
                const std::size_t s = chunkStart[t];
                const std::size_t e = chunkStart[t + 1];
                for (std::size_t i = s; i < e; ++i)
                {
                    values[i] = static_cast<int>(in[i] ^ 0x80000000u);
                }
            });
        }
        for (auto& w : finalWorkers)
        {
            w.join();
        }
    }

    backend = "parallel radix 11/11/10";

    const auto end = std::chrono::high_resolution_clock::now();
    return std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
}

// Parallel LSD radix sort using 8-bit digits (4 passes over 32-bit integers).
//
// Why this beats every other implementation here:
//   - Radix sort is O(4n) vs O(n log n) ≈ O(27n) for comparison sorts — ~7x less work.
//   - 8-bit radix: 256-entry histogram fits in L1 cache (2 KB); 16-bit (65536 entries)
//     would overflow L2 and cause far more cache misses than the extra 2 passes cost.
//   - Parallelism: count phase has zero sharing (each thread owns its histogram slice).
//     Scatter phase is race-free because per-thread offsets are disjoint by construction.
//   - Signed integers are handled by XOR-ing the sign bit, which maps the signed total
//     order onto the unsigned total order without any branches.
long long claude_sort(std::vector<int>& values)
{
    const auto start = std::chrono::high_resolution_clock::now();

    const std::size_t n = values.size();
    if (n <= 1)
    {
        const auto end = std::chrono::high_resolution_clock::now();
        return std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    }

    unsigned int threadCount = std::thread::hardware_concurrency();
    if (threadCount == 0) threadCount = 4;

    constexpr unsigned int RADIX = 8;
    constexpr unsigned int BUCKETS = 1u << RADIX; // 256
    constexpr unsigned int PASSES = 32u / RADIX;  // 4

    // Flip the sign bit so the signed total order matches unsigned comparison:
    //   INT_MIN -> 0,  -1 -> 2^31-1,  0 -> 2^31,  INT_MAX -> 2^32-1
    std::vector<unsigned int> buf(n), aux(n);
    for (std::size_t i = 0; i < n; ++i)
        buf[i] = static_cast<unsigned int>(values[i]) ^ 0x80000000u;

    // Chunk boundaries — distribute remainder elements to first threads
    std::vector<std::size_t> chunkStart(threadCount + 1);
    {
        const std::size_t base = n / threadCount;
        const std::size_t rem = n % threadCount;
        std::size_t pos = 0;
        for (unsigned int t = 0; t < threadCount; ++t)
        {
            chunkStart[t] = pos;
            pos += base + (t < rem ? 1u : 0u);
        }
        chunkStart[threadCount] = n;
    }

    // histFlat[t * BUCKETS + b] = count of digit b in thread t's chunk
    std::vector<std::size_t> histFlat(static_cast<std::size_t>(threadCount) * BUCKETS);

    // offsets[t * BUCKETS + b] = first output index for thread t's bucket-b elements.
    // Computed once per pass via prefix sum; each thread then increments its own slice
    // locally during scatter — no two threads touch the same output position.
    std::vector<std::size_t> offsets(static_cast<std::size_t>(threadCount) * BUCKETS);

    std::vector<std::thread> workers;
    workers.reserve(threadCount);

    for (unsigned int pass = 0; pass < PASSES; ++pass)
    {
        const unsigned int shift = pass * RADIX;

        // --- Parallel count phase: zero sharing, each thread owns its histogram slice ---
        std::fill(histFlat.begin(), histFlat.end(), 0u);
        workers.clear();
        for (unsigned int t = 0; t < threadCount; ++t)
        {
            workers.emplace_back([&, t]() {
                std::size_t* h = histFlat.data() + t * BUCKETS;
                const std::size_t s = chunkStart[t];
                const std::size_t e = chunkStart[t + 1];
                for (std::size_t i = s; i < e; ++i)
                    h[(buf[i] >> shift) & (BUCKETS - 1)]++;
            });
        }
        for (auto& w : workers) w.join();

        // --- Prefix sum: compute per-thread scatter offsets (sequential, O(T*256)) ---
        // Within bucket b the output layout is: [thread0 | thread1 | ... | threadT-1]
        {
            std::size_t globalPos = 0;
            for (unsigned int b = 0; b < BUCKETS; ++b)
            {
                for (unsigned int t = 0; t < threadCount; ++t)
                {
                    offsets[t * BUCKETS + b] = globalPos;
                    globalPos += histFlat[t * BUCKETS + b];
                }
            }
        }

        // --- Parallel scatter phase: each thread writes to a disjoint output region ---
        workers.clear();
        for (unsigned int t = 0; t < threadCount; ++t)
        {
            workers.emplace_back([&, t]() {
                // Stack-local copy so increments don't touch shared memory
                std::size_t pos[BUCKETS];
                for (unsigned int b = 0; b < BUCKETS; ++b)
                    pos[b] = offsets[t * BUCKETS + b];

                const std::size_t s = chunkStart[t];
                const std::size_t e = chunkStart[t + 1];
                for (std::size_t i = s; i < e; ++i)
                {
                    const unsigned int b = (buf[i] >> shift) & (BUCKETS - 1);
                    aux[pos[b]++] = buf[i];
                }
            });
        }
        for (auto& w : workers) w.join();

        std::swap(buf, aux);
    }

    // XOR sign bit back to recover signed integers
    for (std::size_t i = 0; i < n; ++i)
        values[i] = static_cast<int>(buf[i] ^ 0x80000000u);

    const auto end = std::chrono::high_resolution_clock::now();
    return std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
}

// Same parallel LSD radix sort as openmp_sort, but driven by std::thread instead of
// OpenMP so it compiles and runs on any platform (including Linux without OpenMP).
// Uses a simple barrier built from mutex + condition_variable to synchronize threads
// across count / prefix-sum / scatter phases — avoids the thread-creation overhead of
// spawning new threads per phase (like claude_sort does).
long long multithread_sort_ex(std::vector<int>& values)
{
    const auto start = std::chrono::high_resolution_clock::now();

    const std::size_t n = values.size();
    if (n <= 1)
    {
        const auto end = std::chrono::high_resolution_clock::now();
        return std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    }

    unsigned int threadCount = std::thread::hardware_concurrency();
    if (threadCount == 0) threadCount = 4;
    if (threadCount > n) threadCount = static_cast<unsigned int>(n);

    constexpr unsigned int RADIX   = 8;
    constexpr unsigned int BUCKETS = 1u << RADIX; // 256
    constexpr unsigned int PASSES  = 32u / RADIX;  // 4

    std::vector<unsigned int> buf(n), aux(n);

    // Parallel sign-bit flip using std::thread
    {
        std::vector<std::thread> workers;
        workers.reserve(threadCount);
        for (unsigned int t = 0; t < threadCount; ++t)
        {
            workers.emplace_back([&, t]() {
                const std::size_t s = (n * static_cast<std::size_t>(t))       / static_cast<std::size_t>(threadCount);
                const std::size_t e = (n * static_cast<std::size_t>(t + 1))   / static_cast<std::size_t>(threadCount);
                for (std::size_t i = s; i < e; ++i)
                    buf[i] = static_cast<unsigned int>(values[i]) ^ 0x80000000u;
            });
        }
        for (auto& w : workers) w.join();
    }

    // histFlat[t * BUCKETS + b] = count of digit b seen by thread t
    std::vector<std::size_t> histFlat(static_cast<std::size_t>(threadCount) * BUCKETS);
    // offsets[t * BUCKETS + b] = first output index for thread t's bucket-b elements
    std::vector<std::size_t> offsets(static_cast<std::size_t>(threadCount) * BUCKETS);

    // Barrier: reusable spin-free barrier using mutex + condition_variable
    std::mutex barrierMtx;
    std::condition_variable barrierCv;
    unsigned int barrierCount = 0;
    unsigned int barrierGeneration = 0;

    auto barrierWait = [&]() {
        std::unique_lock<std::mutex> lock(barrierMtx);
        const unsigned int gen = barrierGeneration;
        if (++barrierCount == threadCount)
        {
            barrierCount = 0;
            ++barrierGeneration;
            barrierCv.notify_all();
        }
        else
        {
            barrierCv.wait(lock, [&]() { return barrierGeneration != gen; });
        }
    };

    // Each thread lives for all 4 passes; barriers synchronize phases.
    std::vector<std::thread> workers;
    workers.reserve(threadCount);
    for (unsigned int t = 0; t < threadCount; ++t)
    {
        workers.emplace_back([&, t]() {
            const std::size_t s = (n * static_cast<std::size_t>(t))       / static_cast<std::size_t>(threadCount);
            const std::size_t e = (n * static_cast<std::size_t>(t + 1))   / static_cast<std::size_t>(threadCount);

            for (unsigned int pass = 0; pass < PASSES; ++pass)
            {
                const unsigned int shift = pass * RADIX;

                // --- Count phase: each thread writes its own histogram slice ---
                std::size_t* h = histFlat.data() + static_cast<std::size_t>(t) * BUCKETS;
                std::fill(h, h + BUCKETS, std::size_t(0));
                for (std::size_t i = s; i < e; ++i)
                    h[(buf[i] >> shift) & (BUCKETS - 1)]++;

                barrierWait(); // all threads finished counting

                // --- Prefix sum: thread 0 computes scatter offsets for everyone ---
                if (t == 0)
                {
                    std::size_t globalPos = 0;
                    for (unsigned int b = 0; b < BUCKETS; ++b)
                    {
                        for (unsigned int tt = 0; tt < threadCount; ++tt)
                        {
                            offsets[static_cast<std::size_t>(tt) * BUCKETS + b] = globalPos;
                            globalPos += histFlat[static_cast<std::size_t>(tt) * BUCKETS + b];
                        }
                    }
                }

                barrierWait(); // prefix sum complete

                // --- Scatter phase: each thread writes to disjoint output positions ---
                std::size_t pos[BUCKETS];
                for (unsigned int b = 0; b < BUCKETS; ++b)
                    pos[b] = offsets[static_cast<std::size_t>(t) * BUCKETS + b];

                for (std::size_t i = s; i < e; ++i)
                {
                    const unsigned int b = (buf[i] >> shift) & (BUCKETS - 1);
                    aux[pos[b]++] = buf[i];
                }

                barrierWait(); // all threads finished scattering

                // --- Swap: thread 0 swaps buf and aux ---
                if (t == 0)
                    std::swap(buf, aux);

                barrierWait(); // swap visible to all threads
            }
        });
    }
    for (auto& w : workers) w.join();

    // Parallel sign-bit restore
    {
        std::vector<std::thread> finalWorkers;
        finalWorkers.reserve(threadCount);
        for (unsigned int t = 0; t < threadCount; ++t)
        {
            finalWorkers.emplace_back([&, t]() {
                const std::size_t s = (n * static_cast<std::size_t>(t))       / static_cast<std::size_t>(threadCount);
                const std::size_t e = (n * static_cast<std::size_t>(t + 1))   / static_cast<std::size_t>(threadCount);
                for (std::size_t i = s; i < e; ++i)
                    values[i] = static_cast<int>(buf[i] ^ 0x80000000u);
            });
        }
        for (auto& w : finalWorkers) w.join();
    }

    const auto end = std::chrono::high_resolution_clock::now();
    return std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
}

// Same parallel LSD radix sort as claude_sort, but driven by OpenMP instead of std::thread.
// OpenMP maintains a persistent thread pool across parallel regions, eliminating the
// thread-creation/join overhead that claude_sort pays on every pass (8 spawns x 4 passes).
// The histogram and scatter logic is identical; only the threading layer changes.
long long openmp_sort(std::vector<int>& values)
{
    const auto start = std::chrono::high_resolution_clock::now();

#if HAS_OPENMP
    const std::size_t n = values.size();
    if (n <= 1)
    {
        const auto end = std::chrono::high_resolution_clock::now();
        return std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    }

    const int threadCount = omp_get_max_threads();

    constexpr unsigned int RADIX   = 8;
    constexpr unsigned int BUCKETS = 1u << RADIX; // 256
    constexpr unsigned int PASSES  = 32u / RADIX;  // 4

    std::vector<unsigned int> buf(n), aux(n);

    // Parallel sign-bit flip
    #pragma omp parallel for schedule(static) num_threads(threadCount)
    for (std::ptrdiff_t i = 0; i < static_cast<std::ptrdiff_t>(n); ++i)
        buf[i] = static_cast<unsigned int>(values[i]) ^ 0x80000000u;

    // histFlat[t * BUCKETS + b] = count of digit b seen by thread t
    std::vector<std::size_t> histFlat(static_cast<std::size_t>(threadCount) * BUCKETS);
    // offsets[t * BUCKETS + b] = first output index for thread t's bucket-b elements
    std::vector<std::size_t> offsets(static_cast<std::size_t>(threadCount) * BUCKETS);

    for (unsigned int pass = 0; pass < PASSES; ++pass)
    {
        const unsigned int shift = pass * RADIX;

        // --- Parallel count: each thread writes its own histogram slice, no sharing ---
        std::fill(histFlat.begin(), histFlat.end(), 0u);
        #pragma omp parallel num_threads(threadCount)
        {
            const int t   = omp_get_thread_num();
            const int nt  = omp_get_num_threads();
            const std::size_t s = (n * static_cast<std::size_t>(t))      / static_cast<std::size_t>(nt);
            const std::size_t e = (n * static_cast<std::size_t>(t + 1))  / static_cast<std::size_t>(nt);
            std::size_t* h = histFlat.data() + static_cast<std::size_t>(t) * BUCKETS;
            for (std::size_t i = s; i < e; ++i)
                h[(buf[i] >> shift) & (BUCKETS - 1)]++;
        }

        // --- Prefix sum: disjoint output regions per (thread, bucket) pair ---
        {
            std::size_t globalPos = 0;
            for (unsigned int b = 0; b < BUCKETS; ++b)
            {
                for (int t = 0; t < threadCount; ++t)
                {
                    offsets[static_cast<std::size_t>(t) * BUCKETS + b] = globalPos;
                    globalPos += histFlat[static_cast<std::size_t>(t) * BUCKETS + b];
                }
            }
        }

        // --- Parallel scatter: threads write to disjoint positions, zero contention ---
        #pragma omp parallel num_threads(threadCount)
        {
            const int t   = omp_get_thread_num();
            const int nt  = omp_get_num_threads();
            const std::size_t s = (n * static_cast<std::size_t>(t))      / static_cast<std::size_t>(nt);
            const std::size_t e = (n * static_cast<std::size_t>(t + 1))  / static_cast<std::size_t>(nt);

            // Stack-local offset copy so increments stay in registers/L1
            std::size_t pos[BUCKETS];
            for (unsigned int b = 0; b < BUCKETS; ++b)
                pos[b] = offsets[static_cast<std::size_t>(t) * BUCKETS + b];

            for (std::size_t i = s; i < e; ++i)
            {
                const unsigned int b = (buf[i] >> shift) & (BUCKETS - 1);
                aux[pos[b]++] = buf[i];
            }
        }

        std::swap(buf, aux);
    }

    // Parallel sign-bit restore
    #pragma omp parallel for schedule(static) num_threads(threadCount)
    for (std::ptrdiff_t i = 0; i < static_cast<std::ptrdiff_t>(n); ++i)
        values[i] = static_cast<int>(buf[i] ^ 0x80000000u);

#else
    std::cerr << "Error: openmp_sort requires OpenMP. Falling back to std::sort.\n";
    std::sort(values.begin(), values.end());
#endif

    const auto end = std::chrono::high_resolution_clock::now();
    return std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
}

// Pure C + OpenMP parallel LSD radix sort.
// No C++ standard library inside the function body — only malloc/free/memset,
// raw pointers, C-style fixed arrays, and OpenMP pragmas.
long long openmp_c_sort(std::vector<int>& values)
{
    const auto start = std::chrono::high_resolution_clock::now();

#if HAS_OPENMP
    const int n  = (int)values.size();
    if (n <= 1)
    {
        const auto end = std::chrono::high_resolution_clock::now();
        return std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    }

    const int BUCKETS = 256;           /* 2^8 — histogram fits in L1 cache          */
    const int PASSES  = 4;             /* 4 x 8-bit digits cover all 32 bits        */
    const int TC      = omp_get_max_threads();

    unsigned int* buf  = (unsigned int*)malloc((size_t)n * sizeof(unsigned int));
    unsigned int* aux  = (unsigned int*)malloc((size_t)n * sizeof(unsigned int));
    size_t*       hist = (size_t*)malloc((size_t)TC * BUCKETS * sizeof(size_t));
    size_t*       offs = (size_t*)malloc((size_t)TC * BUCKETS * sizeof(size_t));

    if (!buf || !aux || !hist || !offs)
    {
        free(buf); free(aux); free(hist); free(offs);
        std::sort(values.begin(), values.end());
        const auto end = std::chrono::high_resolution_clock::now();
        return std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    }

    /* --- Flip sign bit: maps signed order onto unsigned order ---
     *   INT_MIN -> 0,  -1 -> 2^31-1,  0 -> 2^31,  INT_MAX -> UINT_MAX            */
    #pragma omp parallel for schedule(static) num_threads(TC)
    for (int i = 0; i < n; i++)
        buf[i] = (unsigned int)values[i] ^ 0x80000000u;

    int pass;
    for (pass = 0; pass < PASSES; pass++)
    {
        const unsigned int shift = (unsigned int)pass * 8u;

        /* --- Count phase: each thread counts its own contiguous chunk ----------- */
        memset(hist, 0, (size_t)TC * BUCKETS * sizeof(size_t));

        #pragma omp parallel num_threads(TC)
        {
            int t  = omp_get_thread_num();
            int s  = (int)((long long)n *  t      / TC);
            int e  = (int)((long long)n * (t + 1) / TC);
            size_t* h = hist + (size_t)t * BUCKETS;
            int i;
            for (i = s; i < e; i++)
                h[(buf[i] >> shift) & 0xFFu]++;
        }

        /* --- Prefix sum: compute per-thread scatter offsets (sequential, O(TC*256)) */
        {
            size_t pos = 0;
            int b, t;
            for (b = 0; b < BUCKETS; b++)
            {
                for (t = 0; t < TC; t++)
                {
                    offs[(size_t)t * BUCKETS + b] = pos;
                    pos += hist[(size_t)t * BUCKETS + b];
                }
            }
        }

        /* --- Scatter phase: each thread writes its chunk to disjoint output slots  */
        #pragma omp parallel num_threads(TC)
        {
            int t  = omp_get_thread_num();
            int s  = (int)((long long)n *  t      / TC);
            int e  = (int)((long long)n * (t + 1) / TC);

            /* stack-local copy of this thread's bucket positions */
            size_t pos[256];
            int b, i;
            for (b = 0; b < BUCKETS; b++)
                pos[b] = offs[(size_t)t * BUCKETS + b];

            for (i = s; i < e; i++)
            {
                unsigned int v   = buf[i];
                unsigned int dig = (v >> shift) & 0xFFu;
                aux[pos[dig]++]  = v;
            }
        }

        /* swap pointers — no data copy */
        { unsigned int* tmp = buf; buf = aux; aux = tmp; }
    }

    /* --- Restore sign bit ------------------------------------------------------ */
    #pragma omp parallel for schedule(static) num_threads(TC)
    for (int i = 0; i < n; i++)
        values[i] = (int)(buf[i] ^ 0x80000000u);

    free(offs);
    free(hist);
    free(aux);
    free(buf);

#else
    std::cerr << "Error: openmp_c_sort requires OpenMP. Falling back to std::sort.\n";
    std::sort(values.begin(), values.end());
#endif

    const auto end = std::chrono::high_resolution_clock::now();
    return std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
}

int main()
{
    const std::size_t count = 100000000;
    std::vector<int> values;
    values.reserve(count);

    std::cout << "Sort Test Tool, generating " << count << " random number...\n";

    std::mt19937 rng(42);
    std::uniform_int_distribution<int> dist(
        std::numeric_limits<int>::min(),
        std::numeric_limits<int>::max());

    for (std::size_t i = 0; i < count; ++i)
    {
        values.push_back(dist(rng));
    }

    {
        std::vector<int> v = values;
        const auto ms = openmp_c_sort(v);
        std::cout << "openmp_c_sort time (C + OpenMP):    " << ms << " ms\n";
    }

    {
        std::vector<int> v = values;
        const auto ms = multithread_sort_ex(v);
        std::cout << "multithread_sort_ex time (radix):   " << ms << " ms\n";
    }

    {
        std::vector<int> v = values;
        const auto ms = openmp_sort(v);
        std::cout << "openmp_sort time (parallel radix):  " << ms << " ms\n";
    }

    {
        std::vector<int> v = values;
        const auto ms = claude_sort(v);     //opus 4.6
        std::cout << "claude_sort time (parallel radix):  " << ms << " ms\n";
    }

    {
        std::vector<int> v = values;
        std::string codexBackend;
        const auto ms = codex_sort(v, codexBackend);    //gpt 5.3
        std::cout << "codex_sort time (" << codexBackend << "): " << ms << " ms\n";
    }

    {
        std::vector<int> v = values;
        const auto ms = stl_sort(v);
        std::cout << "stl_sort time:                      " << ms << " ms\n";
    }

    {
        std::vector<int> v = values;
        const auto ms = multithread_sort(v);
        std::cout << "multithread_sort time:              " << ms << " ms\n";
    }

    {
        std::vector<int> v = values;
        const auto ms = parallel_stl_sort(v);
        std::cout << "parallel_stl_sort time:             " << ms << " ms\n";
    }

    {
        std::vector<int> v = values;
        const auto ms = gpu_sort(v);
        std::cout << "gpu_sort time:                      " << ms << " ms\n";
    }

    return 0;
}
// Run program: Ctrl + F5 or Debug > Start Without Debugging menu
// Debug program: F5 or Debug > Start Debugging menu

// Tips for Getting Started:
//   1. Use the Solution Explorer window to add/manage files
//   2. Use the Team Explorer window to connect to source control
//   3. Use the Output window to see build output and other messages
//   4. Use the Error List window to view errors
//   5. Go to Project > Add New Item to create new code files, or Project > Add Existing Item to add existing code files to the project
//   6. In the future, to open this project again, go to File > Open > Project and select the .sln file
