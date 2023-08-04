#define __CL_ENABLE_EXCEPTIONS
#define CL_TARGET_OPENCL_VERSION 300

#include <stdio.h>
#include <math.h>
#include <CL/cl.h>

// OpenCL kernel
const char *kernelSource =                          "\n" \
"#pragma OPENCL EXTENSION cl_khr_fp64 : enable       \n" \
"__kernel void vecAdd( __global double *a,           \n" \
"                      __global double *b,           \n" \
"                      __global double *c,           \n" \
"                      const unsigned int n){        \n" \
"                                                    \n" \
"    // Global Tread ID                              \n" \
"    int id = get_global_id(0);                      \n" \
"                                                    \n" \
"    // Remain in boundaries                         \n" \
"    if (id < n){                                    \n" \
"        c[id] = a[id] + b[id];                      \n" \
"    }                                               \n" \
"}                                                   \n";

int main(){
    unsigned int n = 1000;

    double *h_a, *h_b, *h_c;

    cl_mem d_a, d_b, d_c;
 
    cl_platform_id cpPlatform;
    cl_device_id device_id;
    cl_context context;
    cl_command_queue queue;
    cl_program program;
    cl_kernel kernel;
 
    size_t bytes = n*sizeof(double);

    h_a = (double*)malloc(bytes);
    h_b = (double*)malloc(bytes);
    h_c = (double*)malloc(bytes);
 
    for (size_t i = 0; i < n; ++i){
        h_a[i] = 1;
        h_b[i] = 1;
    }

    size_t globalSize, localSize;
    cl_int err;
 
    // Number of work items in each local work group
    localSize = 64;
 
    // Number of total work items - localSize must be devisor
    globalSize = ceil(n/(float)localSize)*localSize;
 
    // Bind to platform
    err = clGetPlatformIDs(1, &cpPlatform, NULL);
 
    // Get ID for the device
    err = clGetDeviceIDs(cpPlatform, CL_DEVICE_TYPE_GPU, 1, &device_id, NULL);
 
    // Create a context 
    context = clCreateContext(0, 1, &device_id, NULL, NULL, &err);
 
    // Create a command queue
    queue = clCreateCommandQueueWithProperties(context, device_id, 0, &err);
 
    // Create the compute program from the source buffer
    program = clCreateProgramWithSource(context, 1,
                                        (const char **) & kernelSource,
                                        NULL, &err);
 
    // Build the program executable
    clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
 
    // Create the compute kernel in the program we wish to run
    kernel = clCreateKernel(program, "vecAdd", &err);
 
    // Create the input and output arrays in device memory for our calculation
    d_a = clCreateBuffer(context, CL_MEM_READ_ONLY, bytes, NULL, NULL);
    d_b = clCreateBuffer(context, CL_MEM_READ_ONLY, bytes, NULL, NULL);
    d_c = clCreateBuffer(context, CL_MEM_WRITE_ONLY, bytes, NULL, NULL);
 
    // Write our data set into the input array in device memory
    err = clEnqueueWriteBuffer(queue, d_a, CL_TRUE, 0,
                                   bytes, h_a, 0, NULL, NULL);
    err |= clEnqueueWriteBuffer(queue, d_b, CL_TRUE, 0,
                                   bytes, h_b, 0, NULL, NULL);
 
    // Set the arguments to our compute kernel
    err  = clSetKernelArg(kernel, 0, sizeof(cl_mem), &d_a);
    err |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &d_b);
    err |= clSetKernelArg(kernel, 2, sizeof(cl_mem), &d_c);
    err |= clSetKernelArg(kernel, 3, sizeof(unsigned int), &n);
 
    // Execute the kernel over the entire range of the data set 
    err = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &globalSize,
                                 &localSize, 0, NULL, NULL);
 
    // Wait for the command queue to get serviced before reading back results
    clFinish(queue);
 
    // Read the results from the device
    clEnqueueReadBuffer(queue, d_c, CL_TRUE, 0,
                        bytes, h_c, 0, NULL, NULL );
 
    double sum = 0;
    for(size_t i = 0; i < n; ++i){
        sum += h_c[i];
    }
    printf("final result: %f\n", sum);
 
    // release OpenCL resources
    clReleaseMemObject(d_a);
    clReleaseMemObject(d_b);
    clReleaseMemObject(d_c);
    clReleaseProgram(program);
    clReleaseKernel(kernel);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);
 
    //release host memory
    free(h_a);
    free(h_b);
    free(h_c);
 
    return 0;
}
