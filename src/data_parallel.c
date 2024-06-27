#include "darknet.h"
#include "network.h"  // Include network header for function declaration
#include "image.h"
#include <omp.h>  // Include OpenMP header

void data_parallel_dispatch(network *net, image *input_images, int n, int M, int gpu) {
    #pragma omp parallel for num_threads(M)
    for (int i = 0; i < n; i++) {
        int cpu_id = omp_get_thread_num();  // Use omp_get_thread_num from OpenMP

        image resized_image = resize_image(input_images[i], net->w, net->h);

        struct timespec start, end;
        clock_gettime(CLOCK_MONOTONIC, &start);

        if (gpu) {
            network_predict_gpu(net, resized_image.data);  // Ensure this function is declared
        } else {
            network_predict(*net, resized_image.data);  // Dereference net here
        }

        clock_gettime(CLOCK_MONOTONIC, &end);
        double elapsed_time = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;
        printf("CPU %d processed image %d in %f seconds\n", cpu_id, i, elapsed_time);
    }
}
