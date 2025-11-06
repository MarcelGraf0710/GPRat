#ifndef SYCL_UTILS_HPP
#define SYCL_UTILS_HPP

#define WORK_GROUP_SIZE 16

#include "../target.hpp"
#include <sycl/sycl.hpp>
#include <hpx/algorithm.hpp>

namespace gprat
{
    namespace sycl_backend
    {
        // Set float data type throughout the entire sycl namespace
        #ifdef USE_FLOAT
            using real_t = float;    // Set `real_t` to `float`, i.e., single-precision
        #else
            using real_t = double;   // Set `real_t` to `float`, i.e., double-precision
        #endif

        /**
         * @brief BLAS operation types
         *
         * @see cublasOperation_t
         */
        typedef enum BLAS_TRANSPOSE {
            Blas_no_trans = 0,  // CUBLAS_OP_N
            Blas_trans = 1      // CUBLAS_OP_T
        } BLAS_TRANSPOSE;

        /**
         * @brief BLAS side types
         *
         * @see cublasSideMode_t
         */
        typedef enum BLAS_SIDE {
            Blas_left = 0,  // CUBLAS_SIDE_LEFT
            Blas_right = 1  // CUBLAS_SIDE_RIGHT
        } BLAS_SIDE;

        /**
         * @brief Copies a vector from the host to the device using the next SYCL queue of sycl_device.
         *
         * Allocates device memory for the vector and synchronizes the stream after
         * copying the data.
         *
         * @param h_vector The vector to copy from the host
         * @param gpu The GPU target for computations
         *
         * @return A pointer to the copied vector on the device
         */
        inline double *copy_to_device(const std::vector<double> &h_vector, gprat::SYCL_DEVICE &sycl_device)
        {
            double *d_vector;
            sycl::queue queue = sycl_device.next_queue();

            try
            {
                d_vector = sycl::malloc_device<real_t>(h_vector.size(), queue);
                auto copy_process = queue.memcpy(d_vector, h_vector.data(), h_vector.size() * sizeof(real_t));
                copy_process.wait();

            }
            catch (const sycl::exception& e) 
            {
                std::cout << "SYCL exception: " << e.what() << "\n";
            }
            return d_vector;
        }

        /**
         * @brief Frees the device memory allocated in a vector of shared futures.
         *
         * @param vector The vector of shared futures to free
         */
        inline void free(std::vector<hpx::shared_future<double *>> &vector, const sycl::queue &queue)
        {
            // sycl::queue queue = sycl_device.next_queue();

            try
            {
                for (auto &ptr : vector)
                {
                    sycl::free(ptr.get(), queue);
                }
            }
            catch (const sycl::exception& e) 
            {
                std::cout << "SYCL exception: " << e.what() << "\n";
            }
        }
    }
}

#endif // end of SYCL_UTILS_HPP