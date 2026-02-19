#include "gprat_c.hpp"

#include "cpu/gp_functions.hpp"
#include "utils_c.hpp"
#include <cstdio>

#if GPRAT_WITH_CUDA
#include "gpu/cuda/gp_functions.cuh"
#endif

#if GPRAT_WITH_SYCL
#include "gpu/sycl/sycl_gp_functions.hpp"
#endif

// namespace for GPRat library entities
namespace gprat
{

// Constructor of class GP_data ///////////////////////////////////////////////////////////////////
GP_data::GP_data(const std::string &f_path, int n, int n_reg) :
file_path(f_path),
n_samples(n),
n_regressors(n_reg)
{ data = utils::load_data(f_path, n, n_reg - 1); }

// Generic type constructor of class GP ///////////////////////////////////////////////////////////
GP::GP(
    std::vector<double> input,
    std::vector<double> output,
    int n_tiles,
    int n_tile_size,
    int n_regressors,
    std::vector<double> kernel_hyperparams,
    std::vector<bool> trainable_bool,
    std::shared_ptr<Target> target
) :
training_input_(input),
training_output_(output),
n_tiles_(n_tiles),
n_tile_size_(n_tile_size),
trainable_params_(trainable_bool),
target_(target),
n_reg(n_regressors),
kernel_params(kernel_hyperparams[0], kernel_hyperparams[1], kernel_hyperparams[2])
{}

// CPU-type constructor of class GP ///////////////////////////////////////////////////////////////
GP::GP(
    std::vector<double> input,
    std::vector<double> output,
    int n_tiles,
    int n_tile_size,
    int n_regressors,
    std::vector<double> kernel_hyperparams,
    std::vector<bool> trainable_bool
) 
:
training_input_(input),
training_output_(output),
n_tiles_(n_tiles),
n_tile_size_(n_tile_size),
trainable_params_(trainable_bool),
target_(std::make_shared<CPU>()),
n_reg(n_regressors),
kernel_params(kernel_hyperparams[0], kernel_hyperparams[1], kernel_hyperparams[2])
{}

// CUDA-type constructor of class GP //////////////////////////////////////////////////////////////
GP::GP(
    std::vector<double> input,
    std::vector<double> output,
    int n_tiles,
    int n_tile_size,
    int n_regressors,
    std::vector<double> kernel_hyperparams,
    std::vector<bool> trainable_bool,
    int gpu_id,
    int n_streams
) 
:
training_input_(input),
training_output_(output),
n_tiles_(n_tiles),
n_tile_size_(n_tile_size),
trainable_params_(trainable_bool),

#if GPRAT_WITH_CUDA
target_(std::make_shared<CUDA_GPU>(CUDA_GPU(gpu_id, n_streams))),

#else
target_(std::make_shared<CPU>()),

#endif
n_reg(n_regressors),
kernel_params(kernel_hyperparams[0], kernel_hyperparams[1], kernel_hyperparams[2])
{

    #if !GPRAT_WITH_CUDA
    throw std::runtime_error(
        "Cannot create GP object using CUDA for computation. "
        "CUDA is not available because GPRat has been compiled without CUDA. "
        "Remove arguments gpu_id ("
        + std::to_string(gpu_id) + ") and n_streams (" + std::to_string(n_streams)
        + ") to perform computations on the CPU."
    );
    #endif

}


#if GPRAT_WITH_SYCL
// SYCL-type constructor of class GP //////////////////////////////////////////////////////////////
GP::GP(
  std::vector<double> input,
  std::vector<double> output,
  int n_tiles,
  int n_tile_size,
  int n_regressors,
  std::vector<double> kernel_hyperparams,
  std::vector<bool> trainable_bool,
  const DeviceParameters &parameters
) 
:
training_input_(input),
training_output_(output),
n_tiles_(n_tiles),
n_tile_size_(n_tile_size),
trainable_params_(trainable_bool),

#if GPRAT_WITH_SYCL
target_(std::make_shared<SYCL_DEVICE>(parameters)),

#else
target_(std::make_shared<CPU>()),
#endif

n_reg(n_regressors),
kernel_params(kernel_hyperparams[0], kernel_hyperparams[1], kernel_hyperparams[2])
{
    #if !GPRAT_WITH_SYCL
    throw std::runtime_error(
        "Cannot create GP object using SYCL for computation. "
        "SYCL is not available because GPRat has been compiled without SYCL. "
        "Remove argument parameters to perform computations on the CPU.");
    #endif
}
#endif

std::string GP::repr() const
{
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(12);
    oss << "Kernel_Params: [lengthscale=" << kernel_params.lengthscale << ", vertical_lengthscale="
        << kernel_params.vertical_lengthscale << ", noise_variance=" << kernel_params.noise_variance
        << ", n_regressors=" << n_reg << "], Trainable_Params: [trainable_params l=" << trainable_params_[0]
        << ", trainable_params v=" << trainable_params_[1] << ", trainable_params n=" << trainable_params_[2]
        << "], Target: [" << target_->repr() << "], n_tiles=" << n_tiles_ << ", n_tile_size=" << n_tile_size_;
    return oss.str();
}

std::vector<double> GP::get_training_input() const { return training_input_; }

std::vector<double> GP::get_training_output() const { return training_output_; }

// predict ////////////////////////////////////////////////////////////////////////////////////////////////////////////
std::vector<double> GP::predict(const std::vector<double> &test_input, int m_tiles, int m_tile_size)
{
    #if !GPRAT_WITH_SYCL
    return hpx::async([this, &test_input, m_tiles, m_tile_size]()
    {

        #if GPRAT_WITH_CUDA
        // ---- CUDA --------------------------------------------------------------------------------------------------
        if (target_->is_gpu())
        {
            return gpu::predict(
                training_input_,
                training_output_,
                test_input,
                kernel_params,
                n_tiles_,
                n_tile_size_,
                m_tiles,
                m_tile_size,
                n_reg,
                *std::dynamic_pointer_cast<gprat::CUDA_GPU>(target_));
        }
        else
        {
            return cpu::predict(
                training_input_,
                training_output_,
                test_input,
                kernel_params,
                n_tiles_,
                n_tile_size_,
                m_tiles,
                m_tile_size,
                n_reg);
        }
        // ---- !CUDA -------------------------------------------------------------------------------------------------
        #else
        // ---- Host --------------------------------------------------------------------------------------------------
        return cpu::predict(
            training_input_,
            training_output_,
            test_input,
            kernel_params,
            n_tiles_,
            n_tile_size_,
            m_tiles,
            m_tile_size,
            n_reg);
        // ---- !Host -------------------------------------------------------------------------------------------------
        #endif

    }).get();
    #else
        // ---- SYCL --------------------------------------------------------------------------------------------------
        if (!target_->is_cpu())
        {
            return sycl_backend::predict(
                training_input_,
                training_output_,
                test_input,
                kernel_params,
                n_tiles_,
                n_tile_size_,
                m_tiles,
                m_tile_size,
                n_reg,
                *std::dynamic_pointer_cast<gprat::SYCL_DEVICE>(target_));
        }
        else
        {
            return cpu::predict(
                training_input_,
                training_output_,
                test_input,
                kernel_params,
                n_tiles_,
                n_tile_size_,
                m_tiles,
                m_tile_size,
                n_reg);
        }
        // ---- !SYCL -------------------------------------------------------------------------------------------------
    #endif
}

// predict_with_uncertainty ///////////////////////////////////////////////////////////////////////////////////////////
std::vector<std::vector<double>>
GP::predict_with_uncertainty(const std::vector<double> &test_input, int m_tiles, int m_tile_size)
{
    #if !GPRAT_WITH_SYCL
    return hpx::async([this, &test_input, m_tiles, m_tile_size]()
    {
        #if GPRAT_WITH_CUDA
        // ---- CUDA --------------------------------------------------------------------------------------------------
        if (target_->is_gpu())
        {
            return gpu::predict_with_uncertainty(
                training_input_,
                training_output_,
                test_input,
                kernel_params,
                n_tiles_,
                n_tile_size_,
                m_tiles,
                m_tile_size,
                n_reg,
                *std::dynamic_pointer_cast<gprat::CUDA_GPU>(target_));
        }
        else
        {
            return cpu::predict_with_uncertainty(
                training_input_,
                training_output_,
                test_input,
                kernel_params,
                n_tiles_,
                n_tile_size_,
                m_tiles,
                m_tile_size,
                n_reg);
        }
        // ---- !CUDA -------------------------------------------------------------------------------------------------
        #else
        // ---- Host --------------------------------------------------------------------------------------------------
            return cpu::predict_with_uncertainty(
                training_input_,
                training_output_,
                test_input,
                kernel_params,
                n_tiles_,
                n_tile_size_,
                m_tiles,
                m_tile_size,
                n_reg);
        // ---- !Host -------------------------------------------------------------------------------------------------
        #endif

        }).get();
    #else
        // ---- SYCL --------------------------------------------------------------------------------------------------
        if (!target_->is_cpu())
        {
                // ////// MWE STARTS HERE

                // sycl::queue queue_q{sycl::gpu_selector_v};

                // // queue_q.wait();

                // // Example: 3x3 matrix
                // const std::int64_t t = 3;
                // const std::int64_t s = 3;
                // const std::int64_t lda = t; // leading dimension
                // const std::int64_t incx = 1;
                // const std::int64_t incy = 1;

                // // Host data
                // std::vector<double> A = {1.0, 4.0, 7.0,  // column-major layout
                //                         2.0, 5.0, 8.0,
                //                         3.0, 6.0, 9.0};
                // std::vector<double> x = {1.0, 2.0, 3.0};
                // std::vector<double> y = {0.5, 1.0, 1.5};

                // double alpha = 2.0;
                // double beta  = 0.5;

                // // Device buffers
                // double *d_A = sycl::malloc_device<double>(A.size(), queue_q);
                // double *d_x = sycl::malloc_device<double>(x.size(), queue_q);
                // double *d_y = sycl::malloc_device<double>(y.size(), queue_q);

                // // Copy data to device
                // auto event1 = queue_q.memcpy(d_A, A.data(), sizeof(double) * A.size());
                // auto event2 = queue_q.memcpy(d_x, x.data(), sizeof(double) * x.size());
                // auto event3 = queue_q.memcpy(d_y, y.data(), sizeof(double) * y.size());
                // // queue_q.wait();
                // event1.wait();
                // event2.wait();
                // event3.wait();

                // // Perform GEMV: y = alpha * A * x + beta * y
                // sycl::event e = oneapi::math::blas::column_major::gemv(queue_q,
                //                         oneapi::math::transpose::nontrans,
                //                         t,
                //                         s,
                //                         alpha,
                //                         d_A,
                //                         lda,
                //                         d_x,
                //                         incx,
                //                         beta,
                //                         d_y,
                //                         incy);

                // // Wait for computation to finish
                // e.wait();

                // // Copy result back to host
                // queue_q.memcpy(y.data(), d_y, sizeof(double) * y.size()).wait();

                // // Print result
                // std::cout << "Result y:\n";
                // for (double v : y) std::cout << v << " ";
                // std::cout << "\n";

                // // Free device memory
                // sycl::free(d_A, queue_q);
                // sycl::free(d_x, queue_q);
                // sycl::free(d_y, queue_q);

                // ////// MWE ENDS HERE

            return sycl_backend::predict_with_uncertainty(
                training_input_,
                training_output_,
                test_input,
                kernel_params,
                n_tiles_,
                n_tile_size_,
                m_tiles,
                m_tile_size,
                n_reg,
                *std::dynamic_pointer_cast<gprat::SYCL_DEVICE>(target_));
        }
        else
        {
            return cpu::predict_with_uncertainty(
                training_input_,
                training_output_,
                test_input,
                kernel_params,
                n_tiles_,
                n_tile_size_,
                m_tiles,
                m_tile_size,
                n_reg);
        }
        // ---- !SYCL -------------------------------------------------------------------------------------------------
    #endif
}

// predict_with_full_cov //////////////////////////////////////////////////////////////////////////////////////////////
std::vector<std::vector<double>>
GP::predict_with_full_cov(const std::vector<double> &test_input, int m_tiles, int m_tile_size)
{
    #if !GPRAT_WITH_SYCL
    return hpx::async([this, &test_input, m_tiles, m_tile_size]()
    {
        #if GPRAT_WITH_CUDA
        // ---- CUDA --------------------------------------------------------------------------------------------------
        if (target_->is_gpu())
        {
            return gpu::predict_with_full_cov(
                training_input_,
                training_output_,
                test_input,
                kernel_params,
                n_tiles_,
                n_tile_size_,
                m_tiles,
                m_tile_size,
                n_reg,
                *std::dynamic_pointer_cast<gprat::CUDA_GPU>(target_));
        }
        else
        {
            return cpu::predict_with_full_cov(
                training_input_,
                training_output_,
                test_input,
                kernel_params,
                n_tiles_,
                n_tile_size_,
                m_tiles,
                m_tile_size,
                n_reg);
        }
        // ---- !CUDA -------------------------------------------------------------------------------------------------
        #else
        // ---- Host --------------------------------------------------------------------------------------------------
        return cpu::predict_with_full_cov(
            training_input_,
            training_output_,
            test_input,
            kernel_params,
            n_tiles_,
            n_tile_size_,
            m_tiles,
            m_tile_size,
            n_reg);
        // ---- !Host -------------------------------------------------------------------------------------------------
        #endif

    }).get();
    #else
        // ---- SYCL --------------------------------------------------------------------------------------------------
        if (!target_->is_cpu())
        {
            return sycl_backend::predict_with_full_cov(
                training_input_,
                training_output_,
                test_input,
                kernel_params,
                n_tiles_,
                n_tile_size_,
                m_tiles,
                m_tile_size,
                n_reg,
                *std::dynamic_pointer_cast<gprat::SYCL_DEVICE>(target_));
        }
        else
        {
            return cpu::predict_with_full_cov(
                training_input_,
                training_output_,
                test_input,
                kernel_params,
                n_tiles_,
                n_tile_size_,
                m_tiles,
                m_tile_size,
                n_reg);
        }
        // ---- !SYCL -------------------------------------------------------------------------------------------------
        #endif
}

// optimize ///////////////////////////////////////////////////////////////////////////////////////////////////////////
std::vector<double> GP::optimize(const gprat_hyper::AdamParams &adam_params)
{
    return hpx::async([this, &adam_params]()
    {
    #if GPRAT_WITH_CUDA || GPRAT_WITH_SYCL
        if (target_->is_gpu())
        {
            std::cerr << "GP::optimze_step has not been implemented for the GPU.\n"
                        << "Instead, this operation executes the CPU implementation." << std::endl;
        }
    #endif
        return cpu::optimize(
            training_input_,
            training_output_,
            n_tiles_,
            n_tile_size_,
            n_reg,
            adam_params,
            kernel_params,
            trainable_params_);
    }).get();
}

// optimize_step //////////////////////////////////////////////////////////////////////////////////////////////////////
double GP::optimize_step(gprat_hyper::AdamParams &adam_params, int iter)
{
    return hpx::async([this, &adam_params, iter]()
    {
        #if GPRAT_WITH_CUDA || GPRAT_WITH_SYCL
        if (target_->is_gpu())
        {
            std::cerr << "GP::optimze_step has not been implemented for the GPU.\n"
                        << "Instead, this operation executes the CPU implementation." << std::endl;
        }

        #endif
        return cpu::optimize_step(
            training_input_,
            training_output_,
            n_tiles_,
            n_tile_size_,
            n_reg,
            adam_params,
            kernel_params,
            trainable_params_,
            iter);
    }).get();
}

// calculate_loss /////////////////////////////////////////////////////////////////////////////////////////////////////
double GP::calculate_loss()
{
    return hpx::async([this]()
    {
        #if GPRAT_WITH_CUDA
        if (target_->is_gpu())
        {
            return gpu::compute_loss(
                training_input_,
                training_output_,
                kernel_params,
                n_tiles_,
                n_tile_size_,
                n_reg,
                *std::dynamic_pointer_cast<gprat::CUDA_GPU>(target_));
        }
        else
        {
            return cpu::compute_loss(
                training_input_, training_output_, kernel_params, n_tiles_, n_tile_size_, n_reg);
        }

        #elif GPRAT_WITH_SYCL
        if (!target_->is_cpu())
        {
            return sycl_backend::compute_loss(
                training_input_,
                training_output_,
                kernel_params,
                n_tiles_,
                n_tile_size_,
                n_reg,
                *std::dynamic_pointer_cast<gprat::SYCL_DEVICE>(target_));
        }
        else
        {
            return cpu::compute_loss(
                training_input_, training_output_, kernel_params, n_tiles_, n_tile_size_, n_reg);
        }

        #else
                   return cpu::compute_loss(
                       training_input_, training_output_, kernel_params, n_tiles_, n_tile_size_, n_reg);
        #endif

    }).get();
}

// cholesky ///////////////////////////////////////////////////////////////////////////////////////////////////////////
std::vector<std::vector<double>> GP::cholesky()
{
    #if !GPRAT_WITH_SYCL
    return hpx::async([this]()
    {
        #if GPRAT_WITH_CUDA
        if (target_->is_gpu())
        {
            return gpu::cholesky(
                training_input_,
                kernel_params,
                n_tiles_,
                n_tile_size_,
                n_reg,
                *std::dynamic_pointer_cast<gprat::CUDA_GPU>(target_));
        }
        else
        {
            return cpu::cholesky(training_input_, kernel_params, n_tiles_, n_tile_size_, n_reg);
        }
        #else
            return cpu::cholesky(training_input_, kernel_params, n_tiles_, n_tile_size_, n_reg);
        #endif
    }).get();
    #else
    
        if (!target_->is_cpu())
        {
            return sycl_backend::cholesky(
                training_input_,
                kernel_params,
                n_tiles_,
                n_tile_size_,
                n_reg,
                *std::dynamic_pointer_cast<gprat::SYCL_DEVICE>(target_));
        }
        else
        {
            return cpu::cholesky(training_input_, kernel_params, n_tiles_, n_tile_size_, n_reg);
        }

    #endif
}

}  // namespace gprat
