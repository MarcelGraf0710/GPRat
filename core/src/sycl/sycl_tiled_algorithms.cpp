#include "sycl/sycl_tiled_algorithms.hpp"

#include "sycl/adapter_onemath.hpp"
#include "sycl/sycl_gp_optimizer.hpp"
#include "sycl/sycl_gp_uncertainty.hpp"
#include <hpx/algorithm.hpp>

namespace gprat::sycl_backend
{

// Tiled Cholesky Algorithm

void right_looking_cholesky_tiled(
    std::vector<hpx::shared_future<double *>> &ft_tiles,
    const std::size_t n_tile_size,
    const std::size_t n_tiles,
    gprat::SYCL_DEVICE &sycl_device
)
{
    std::cout << "[sycl_tiled_algorithms.cpp] [right_looking_cholesky_tiled] : Entering \n";

    sycl::queue queue;

    std::cout << "[sycl_tiled_algorithms.cpp] [right_looking_cholesky_tiled] : Entering loop ALPHA \n";

    for (std::size_t k = 0; k < n_tiles; ++k)
    {
        std::cout << "[sycl_tiled_algorithms.cpp] [right_looking_cholesky_tiled] : LOOP ALPHA : ITERATION " << k << " \n";

        queue = sycl_device.next_queue();

        std::cout << "[sycl_tiled_algorithms.cpp] [right_looking_cholesky_tiled] : POTRF with index " << static_cast<std::size_t>(k) * n_tiles + static_cast<std::size_t>(k) << "\n";

        // ft_tiles[static_cast<std::size_t>(k) * n_tiles + static_cast<std::size_t>(k)] = hpx::dataflow(
        //     [&queue, &ft_tiles, n_tile_size, n_tiles, k](double* pointer){ return potrf(queue, pointer, n_tile_size); },
        //     hpx::unwrap(ft_tiles[static_cast<std::size_t>(k) * n_tiles + static_cast<std::size_t>(k)])
        // );

        ft_tiles[static_cast<std::size_t>(k) * n_tiles + static_cast<std::size_t>(k)] = hpx::make_ready_future<double *>(
            potrf(queue, ft_tiles[static_cast<std::size_t>(k) * n_tiles + static_cast<std::size_t>(k)].get(), n_tile_size)
        );

        std::cout << "[sycl_tiled_algorithms.cpp] [right_looking_cholesky_tiled] : POTRF submitted, waiting for result \n";

        // NOTE: The result is immediately needed by TRSM. Also TRSM may throw
        // an error otherwise.
        // ft_tiles[static_cast<std::size_t>(k) * n_tiles + static_cast<std::size_t>(k)].get();

        std::cout << "[sycl_tiled_algorithms.cpp] [right_looking_cholesky_tiled] : Entering loop BETA \n";

        for (std::size_t m = k + 1; m < n_tiles; ++m)
        {
            std::cout << "[sycl_tiled_algorithms.cpp] [right_looking_cholesky_tiled] : LOOP BETA : ITERATION " << m << " \n";
            queue = sycl_device.next_queue();

            std::cout << "[sycl_tiled_algorithms.cpp] [right_looking_cholesky_tiled] : TRSM \n";

            ft_tiles[m * n_tiles + k] = hpx::make_ready_future<double *>(
                trsm(
                queue,
                ft_tiles[static_cast<std::size_t>(k) * n_tiles + static_cast<std::size_t>(k)].get(),
                ft_tiles[m * n_tiles + k].get(),
                n_tile_size,
                n_tile_size,
                oneapi::math::transpose::trans,
                oneapi::math::side::right)
            );

            // // TRSM
            // ft_tiles[m * n_tiles + k] = hpx::dataflow(
            //     [&queue, &ft_tiles, n_tile_size, n_tiles, k, m]{ return trsm(
            //     queue,
            //     ft_tiles[static_cast<std::size_t>(k) * n_tiles + static_cast<std::size_t>(k)].get(),
            //     ft_tiles[m * n_tiles + k].get(),
            //     n_tile_size,
            //     n_tile_size,
            //     oneapi::math::transpose::trans,
            //     oneapi::math::side::right); }
            // );
        }

        // std::cout << "[sycl_tiled_algorithms.cpp] [right_looking_cholesky_tiled] : Entering loop GAMMA \n";

        for (std::size_t m = k + 1; m < n_tiles; ++m)
        {
            // std::cout << "[sycl_tiled_algorithms.cpp] [right_looking_cholesky_tiled] : LOOP GAMMA : ITERATION " << m << " \n";
            queue = sycl_device.next_queue();

            std::cout << "[sycl_tiled_algorithms.cpp] [right_looking_cholesky_tiled] : SYRK \n";

            // SYRK
            ft_tiles[m * n_tiles + m] =
                hpx::dataflow([&queue, &ft_tiles, n_tile_size, n_tiles, k, m]() {return syrk(queue, ft_tiles[m * n_tiles + k].get(), ft_tiles[m * n_tiles + m].get(), n_tile_size);});

            // std::cout << "[sycl_tiled_algorithms.cpp] [right_looking_cholesky_tiled] : Entering loop DELTA \n";

            for (std::size_t n = k + 1; n < m; ++n)
            {
                // std::cout << "[sycl_tiled_algorithms.cpp] [right_looking_cholesky_tiled] : LOOP DELTA : ITERATION " << n << " \n";
                queue = sycl_device.next_queue();

                std::cout << "[sycl_tiled_algorithms.cpp] [right_looking_cholesky_tiled] : GEMM \n";

                // GEMM
                ft_tiles[m * n_tiles + n] = hpx::dataflow(
                    [&queue, &ft_tiles, n_tile_size, n_tiles, k, m, n]() { return gemm(
                    queue,
                    ft_tiles[m * n_tiles + k].get(),
                    ft_tiles[n * n_tiles + k].get(),
                    ft_tiles[m * n_tiles + n].get(),
                    n_tile_size,
                    n_tile_size,
                    n_tile_size,
                    oneapi::math::transpose::nontrans,
                    oneapi::math::transpose::trans); });
            }
        }
    }

    std::cout << "[sycl_tiled_algorithms.cpp] [right_looking_cholesky_tiled] : Leaving \n";
}

// Tiled Triangular Solve Algorithms

void forward_solve_tiled(
    std::vector<hpx::shared_future<double *>> &ft_tiles,
    std::vector<hpx::shared_future<double *>> &ft_rhs,
    const std::size_t n_tile_size,
    const std::size_t n_tiles,
    gprat::SYCL_DEVICE &sycl_device
)
{
    sycl::queue queue;

    for (std::size_t k = 0; k < n_tiles; ++k)
    {
        queue = sycl_device.next_queue();

        // TRSM: Solve L * x = a
        ft_rhs[k] = hpx::dataflow(
            [&]() { return trsv( 
            queue, 
            ft_tiles[static_cast<std::size_t>(k) * n_tiles + static_cast<std::size_t>(k)], 
            ft_rhs[k], 
            n_tile_size, 
            oneapi::math::transpose::nontrans
        ); } );

        for (std::size_t m = k + 1; m < n_tiles; ++m)
        {
            queue = sycl_device.next_queue();

            // GEMV: b = b - A * a
            ft_rhs[m] = hpx::dataflow(
                [&]() { return gemv(
                queue,
                ft_tiles[m * n_tiles + k],
                ft_rhs[k],
                ft_rhs[m],
                n_tile_size,
                n_tile_size,
                -1,
                oneapi::math::transpose::nontrans
            ); } );
        }
    }
}

void backward_solve_tiled(
    std::vector<hpx::shared_future<double *>> &ft_tiles,              
    std::vector<hpx::shared_future<double *>> &ft_rhs,              
    const std::size_t n_tile_size,            
    const std::size_t n_tiles,          
    gprat::SYCL_DEVICE &sycl_device
)
{
    sycl::queue queue;
    // NOTE: The loops traverse backwards. Its last comparisons require the
    // usage negative numbers. Therefore they use signed int instead of the
    // unsigned std::size_t.

    for (int k = static_cast<int>(n_tiles) - 1; k >= 0; k--)
    {
        queue = sycl_device.next_queue();

        // TRSM: Solve L^T * x = a
        ft_rhs[static_cast<std::size_t>(k)] = hpx::dataflow(
            [&]() { return trsv(
            queue, 
            ft_tiles[static_cast<std::size_t>(static_cast<std::size_t>(k) * n_tiles + static_cast<std::size_t>(k))], 
            ft_rhs[static_cast<std::size_t>(k)], 
            n_tile_size, 
            oneapi::math::transpose::trans
        ); } );

        for (int m = k - 1; m >= 0; m--)
        {
            queue = sycl_device.next_queue();

            // GEMV: b = b - A^T * a
            ft_rhs[static_cast<std::size_t>(m)] = hpx::dataflow(
                [&]() { return gemv(
                queue,
                ft_tiles[static_cast<std::size_t>(k) * n_tiles + static_cast<std::size_t>(m)],
                ft_rhs[static_cast<std::size_t>(k)],
                ft_rhs[static_cast<std::size_t>(m)],
                n_tile_size,
                n_tile_size,
                -1,
                oneapi::math::transpose::trans
            ); } );
        }
    }
}

void forward_solve_tiled_matrix(
    std::vector<hpx::shared_future<double *>> &ft_tiles,
    std::vector<hpx::shared_future<double *>> &ft_rhs,
    const std::size_t n_tile_size,
    const std::size_t m_tile_size,
    const std::size_t n_tiles,
    const std::size_t m_tiles,
    gprat::SYCL_DEVICE &sycl_device
)
{
    sycl::queue queue;

    for (std::size_t c = 0; c < m_tiles; ++c)
    {
        for (std::size_t k = 0; k < n_tiles; ++k)
        {
            queue = sycl_device.next_queue();

            // TRSM: solve L * X = A
            ft_rhs[static_cast<std::size_t>(static_cast<std::size_t>(k * m_tiles + c))] = hpx::dataflow(
                [&]() {
                return trsm(
                queue,
                ft_tiles[static_cast<std::size_t>(k) * n_tiles + static_cast<std::size_t>(k)].get(),
                ft_rhs[static_cast<std::size_t>(k * m_tiles + c)].get(),
                n_tile_size,
                m_tile_size,
                oneapi::math::transpose::nontrans,
                oneapi::math::side::left
                ); });

            for (std::size_t m = k + 1; m < n_tiles; ++m)
            {
                queue = sycl_device.next_queue();

                // GEMM: C = C - A * B
                ft_rhs[m * m_tiles + c] = hpx::dataflow(
                    [&]() { return gemm(
                    queue,
                    ft_tiles[m * n_tiles + k].get(),
                    ft_rhs[static_cast<std::size_t>(k * m_tiles + c)].get(),
                    ft_rhs[m * m_tiles + c].get(),
                    n_tile_size,
                    m_tile_size,
                    n_tile_size,
                    oneapi::math::transpose::nontrans,
                    oneapi::math::transpose::nontrans
                ); } );
            }
        }
    }
}

void backward_solve_tiled_matrix(
    std::vector<hpx::shared_future<double *>> &ft_tiles,
    std::vector<hpx::shared_future<double *>> &ft_rhs,
    const std::size_t n_tile_size,
    const std::size_t m_tile_size,
    const std::size_t n_tiles,
    const std::size_t m_tiles,
    gprat::SYCL_DEVICE &sycl_device
)
{
    sycl::queue queue;

    for (std::size_t c = 0; c < m_tiles; ++c)
    {
        for (std::size_t k = 0; k < n_tiles; ++k)
        {
            queue = sycl_device.next_queue();

            // TRSM: solve L^T * X = A
            ft_rhs[static_cast<std::size_t>(k * m_tiles + c)] = hpx::dataflow(
                [&]() {
                return trsm(
                queue,
                ft_tiles[static_cast<std::size_t>(k) * n_tiles + static_cast<std::size_t>(k)].get(),
                ft_rhs[static_cast<std::size_t>(k * m_tiles + c)].get(),
                n_tile_size,
                m_tile_size,
                oneapi::math::transpose::trans,
                oneapi::math::side::left
            ); } );

            for (std::size_t m = 0; m < k; ++m)
            {
                queue = sycl_device.next_queue();

                // GEMM: C = C - A^T * B
                ft_rhs[m * m_tiles + c] = hpx::dataflow(
                    [&]() {
                    return gemm(
                    queue,
                    ft_tiles[k * n_tiles + m].get(),
                    ft_rhs[static_cast<std::size_t>(k * m_tiles + c)].get(),
                    ft_rhs[m * m_tiles + c].get(),
                    n_tile_size,
                    m_tile_size,
                    n_tile_size,
                    oneapi::math::transpose::trans,
                    oneapi::math::transpose::nontrans
                ); } );
            }
        }
    }
}

void matrix_vector_tiled(
    std::vector<hpx::shared_future<double *>> &ft_tiles,
    std::vector<hpx::shared_future<double *>> &ft_vector,
    std::vector<hpx::shared_future<double *>> &ft_rhs,
    const std::size_t N_row,
    const std::size_t N_col,
    const std::size_t n_tiles,
    const std::size_t m_tiles,
    gprat::SYCL_DEVICE &sycl_device
)
{
    sycl::queue queue;

    for (std::size_t k = 0; k < m_tiles; ++k)
    {
        for (std::size_t m = 0; m < n_tiles; ++m)
        {
            queue = sycl_device.next_queue();

            ft_rhs[k] = hpx::dataflow(
                [&]() {
                return gemv(
                queue,
                ft_tiles[k * n_tiles + m],
                ft_vector[m],
                ft_rhs[k],
                N_row,
                N_col,
                1,
                oneapi::math::transpose::nontrans
            ); } );
        }
    }
}

void symmetric_matrix_matrix_diagonal_tiled(
    std::vector<hpx::shared_future<double *>> &ft_tCC_tiles,
    std::vector<hpx::shared_future<double *>> &ft_inter_tiles,
    const std::size_t n_tile_size,
    const std::size_t m_tile_size,
    const std::size_t n_tiles,
    const std::size_t m_tiles,
    gprat::SYCL_DEVICE &sycl_device
)
{
    sycl::queue queue;

    for (std::size_t i = 0; i < m_tiles; ++i)
    {
        for (std::size_t n = 0; n < n_tiles; ++n)
        {
            queue = sycl_device.next_queue();

            // Compute inner product to obtain diagonal elements of
            // (K_MxN * (K^-1_NxN * K_NxM))
            ft_inter_tiles[i] = hpx::dataflow(
                [&]() {
                return dot_diag_syrk(
                queue,
                ft_tCC_tiles[n * m_tiles + i],
                ft_inter_tiles[i],
                n_tile_size,
                m_tile_size
            ); } );
        }
    }
}

void compute_gemm_of_invK_y(
    std::vector<hpx::shared_future<double *>> &ft_invK,
    std::vector<hpx::shared_future<double *>> &ft_y,
    std::vector<hpx::shared_future<double *>> &ft_alpha,
    const std::size_t n_tile_size,
    const std::size_t n_tiles,
    gprat::SYCL_DEVICE &sycl_device
)
{
    sycl::queue queue;

    for (std::size_t i = 0; i < n_tiles; ++i)
    {
        for (std::size_t j = 0; j < n_tiles; ++j)
        {
            queue = sycl_device.next_queue();

            ft_alpha[i] = hpx::dataflow(
                [&]() {
                return gemv(
                queue,
                ft_invK[i * n_tiles + j],
                ft_y[j],
                ft_alpha[i],
                n_tile_size,
                n_tile_size,
                1,
                oneapi::math::transpose::nontrans
            ); } );
        }
    }
}

hpx::shared_future<double> compute_loss_tiled(
    std::vector<hpx::shared_future<double *>> &ft_tiles,
    std::vector<hpx::shared_future<double *>> &ft_alpha,
    std::vector<hpx::shared_future<double *>> &ft_y,
    const std::size_t n_tile_size,
    const std::size_t n_tiles,
    gprat::SYCL_DEVICE &sycl_device
)
{
    std::vector<hpx::shared_future<double>> loss_tiled(n_tiles);

    for (std::size_t k = 0; k < n_tiles; k++)
    {
        loss_tiled[k] =
            hpx::dataflow([&](){ return compute_loss(ft_tiles[static_cast<std::size_t>(k) * n_tiles + static_cast<std::size_t>(k)], ft_alpha[k], ft_y[k], n_tile_size, std::ref(sycl_device)); } );
    }

    return hpx::dataflow(&add_losses, loss_tiled, n_tile_size, n_tiles);
}

void symmetric_matrix_matrix_tiled(
    std::vector<hpx::shared_future<double *>> &ft_tCC_tiles,
    std::vector<hpx::shared_future<double *>> &ft_priorK,
    const std::size_t n_tile_size,
    const std::size_t m_tile_size,
    const std::size_t n_tiles,
    const std::size_t m_tiles,
    gprat::SYCL_DEVICE &sycl_device
)
{
    sycl::queue queue;

    for (std::size_t c = 0; c < m_tiles; ++c)
    {
        for (std::size_t k = 0; k < m_tiles; ++k)
        {
            for (std::size_t m = 0; m < n_tiles; ++m)
            {
                queue = sycl_device.next_queue();

                // GEMM:  C = C - A^T * B
                ft_priorK[c * m_tiles + k] = hpx::dataflow(
                    [&]() {
                    return gemm(
                    queue,
                    ft_tCC_tiles[m * m_tiles + c].get(),
                    ft_tCC_tiles[m * m_tiles + k].get(),
                    ft_priorK[c * m_tiles + k].get(),
                    n_tile_size,
                    m_tile_size,
                    m_tile_size,
                    oneapi::math::transpose::trans,
                    oneapi::math::transpose::nontrans
                ); } );
            }
        }
    }
}

void vector_difference_tiled(
    std::vector<hpx::shared_future<double *>> &ft_priorK,
    std::vector<hpx::shared_future<double *>> &ft_inter,
    std::vector<hpx::shared_future<double *>> &ft_vector,
    const std::size_t m_tile_size,
    const std::size_t m_tiles,
    gprat::SYCL_DEVICE &sycl_device
)
{
    for (std::size_t i = 0; i < m_tiles; i++)
    {
        ft_vector[i] = hpx::dataflow([&]() { return diag_posterior(ft_priorK[i], ft_inter[i], m_tile_size, std::ref(sycl_device)); } );
    }
}

void matrix_diagonal_tiled(
    std::vector<hpx::shared_future<double *>> &ft_priorK,
    std::vector<hpx::shared_future<double *>> &ft_vector,
    const std::size_t m_tile_size,
    const std::size_t m_tiles,
    gprat::SYCL_DEVICE &sycl_device
)
{
    for (std::size_t i = 0; i < m_tiles; i++)
    {
        ft_vector[i] = hpx::dataflow([&]() { return diag_tile(ft_priorK[i * m_tiles + i], m_tile_size, std::ref(sycl_device)); } );
    }
}

void update_grad_K_tiled_mkl(
    std::vector<hpx::shared_future<double *>> &ft_tiles,
    const std::vector<hpx::shared_future<double *>> &ft_v1,
    const std::vector<hpx::shared_future<double *>> &ft_v2,
    const std::size_t n_tile_size,
    const std::size_t n_tiles,
    gprat::SYCL_DEVICE &sycl_device
)
{
    sycl::queue queue;

    for (std::size_t i = 0; i < n_tiles; ++i)
    {
        for (std::size_t j = 0; j < n_tiles; ++j)
        {
            queue = sycl_device.next_queue();

            ft_tiles[i * n_tiles + j] =
                hpx::dataflow([&]() { return ger(queue, ft_tiles[i * n_tiles + j], ft_v1[i], ft_v2[j], n_tile_size); } );
        }
    }
}

static double update_hyperparameter(
    // const std::vector<hpx::shared_future<double *>> &ft_invK,
    // const std::vector<hpx::shared_future<double *>> &ft_gradparam,
    // const std::vector<hpx::shared_future<double *>> &ft_alpha,
    // double &hyperparameter,  // lengthscale or vertical-lengthscale
    // gprat_hyper::SEKParams sek_params,
    // gprat_hyper::AdamParams adam_params,
    // const std::size_t n_tile_size,
    // const std::size_t n_tiles,
    // std::vector<hpx::shared_future<double>> &m_T,
    // std::vector<hpx::shared_future<double>> &v_T,
    // const std::vector<hpx::shared_future<double>> &beta1_T,
    // const std::vector<hpx::shared_future<double>> &beta2_T,
    // int iter,
    // int param_idx,  // 0 for lengthscale, 1 for vertical-lengthscale
    // gprat::SYCL_DEVICE &sycl_device
)
{
    throw std::logic_error("Function not implemented for GPU");
    // return 0;
}

double update_lengthscale(
    // const std::vector<hpx::shared_future<double *>> &ft_invK,
    // const std::vector<hpx::shared_future<double *>> &ft_gradparam,
    // const std::vector<hpx::shared_future<double *>> &ft_alpha,
    // gprat_hyper::SEKParams sek_params,
    // gprat_hyper::AdamParams adam_params,
    // const std::size_t n_tile_size,
    // const std::size_t n_tiles,
    // std::vector<hpx::shared_future<double>> &m_T,
    // std::vector<hpx::shared_future<double>> &v_T,
    // const std::vector<hpx::shared_future<double>> &beta1_T,
    // const std::vector<hpx::shared_future<double>> &beta2_T,
    // int iter,
    // gprat::SYCL_DEVICE &sycl_device
)
{
    return update_hyperparameter(
        // ft_invK,
        // ft_gradparam,
        // ft_alpha,
        // sek_params.lengthscale,
        // sek_params,
        // adam_params,
        // n_tile_size,
        // n_tiles,
        // m_T,
        // v_T,
        // beta1_T,
        // beta2_T,
        // iter,
        // 0,
        // sycl_device
    );
}

double update_vertical_lengthscale(
    // const std::vector<hpx::shared_future<double *>> &ft_invK,
    // const std::vector<hpx::shared_future<double *>> &ft_gradparam,
    // const std::vector<hpx::shared_future<double *>> &ft_alpha,
    // gprat_hyper::SEKParams sek_params,
    // gprat_hyper::AdamParams adam_params,
    // const std::size_t n_tile_size,
    // const std::size_t n_tiles,
    // std::vector<hpx::shared_future<double>> &m_T,
    // std::vector<hpx::shared_future<double>> &v_T,
    // const std::vector<hpx::shared_future<double>> &beta1_T,
    // const std::vector<hpx::shared_future<double>> &beta2_T,
    // int iter,
    // gprat::SYCL_DEVICE &sycl_device
)
{
    return update_hyperparameter(
        // ft_invK,
        // ft_gradparam,
        // ft_alpha,
        // sek_params.vertical_lengthscale,
        // sek_params,
        // adam_params,
        // n_tile_size,
        // n_tiles,
        // m_T,
        // v_T,
        // beta1_T,
        // beta2_T,
        // iter,
        // 1,
        // sycl_device
    );
}

double update_noise_variance(
    // const std::vector<hpx::shared_future<double *>> &ft_invK,
    // const std::vector<hpx::shared_future<double *>> &ft_alpha,
    // gprat_hyper::SEKParams sek_params,
    // gprat_hyper::AdamParams adam_params,
    // const std::size_t n_tile_size,
    // const std::size_t n_tiles,
    // std::vector<hpx::shared_future<double>> &m_T,
    // std::vector<hpx::shared_future<double>> &v_T,
    // const std::vector<hpx::shared_future<double>> &beta1_T,
    // const std::vector<hpx::shared_future<double>> &beta2_T,
    // int iter,
    // gprat::SYCL_DEVICE &sycl_device
)
{
    throw std::logic_error("Function not implemented for GPU");
    // return 0;
}

}  // end of namespace sycl_backend
