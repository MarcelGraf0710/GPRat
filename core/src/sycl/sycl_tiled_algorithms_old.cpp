#include "sycl/sycl_tiled_algorithms.hpp"

#include "sycl/adapter_onemath.hpp"
#include "sycl/sycl_gp_optimizer.hpp"
#include "sycl/sycl_gp_uncertainty.hpp"
#include <hpx/algorithm.hpp>

namespace sycl_backend
{

// Tiled Cholesky Algorithm

void right_looking_cholesky_tiled(
    std::vector<hpx::shared_future<double *>> &ft_tiles,
    const std::size_t n_tile_size,
    const std::size_t n_tiles,
    gprat::SYCL_DEVICE &sycl_device
)
{
    sycl::queue queue;

    for (std::size_t k = 0; k < n_tiles; ++k)
    {
        queue = sycl_device.next_queue();

        // POTRF
        // ft_tiles[k * n_tiles + k] = hpx::dataflow(
        //     hpx::annotated_function(&potrf, "Cholesky POTRF"),
        //     queue,
        //     ft_tiles[k * n_tiles + k],
        //     n_tile_size
        // );
        ft_tiles[k * n_tiles + k] =  hpx::dataflow(
            [&queue, n_tile_size](hpx::shared_future<double*> f_tile)
            { return potrf(queue, f_tile, n_tile_size); },
            ft_tiles[k * n_tiles + k]
        );

        // NOTE: The result is immediately needed by TRSM. Also TRSM may throw
        // an error otherwise.
        ft_tiles[k * n_tiles + k].get();

        for (std::size_t m = k + 1; m < n_tiles; ++m)
        {
            queue = sycl_device.next_queue();

            // TRSM
            // ft_tiles[m * n_tiles + k] = hpx::dataflow(
            //     &trsm,
            //     queue,
            //     ft_tiles[k * n_tiles + k],
            //     ft_tiles[m * n_tiles + k],
            //     n_tile_size,
            //     n_tile_size,
            //     oneapi::math::transpose::trans,
            //     oneapi::math::side::right
            // );
            ft_tiles[m * n_tiles + k] = hpx::dataflow(
                [&queue, n_tile_size](
                    hpx::shared_future<double *> f_A,
                    hpx::shared_future<double *> f_B
                )
                { return trsm(queue, f_A, f_B, n_tile_size, n_tile_size, oneapi::math::transpose::trans, oneapi::math::side::right); },
                ft_tiles[k * n_tiles + k], ft_tiles[m * n_tiles + k]
            );
        }

        for (std::size_t m = k + 1; m < n_tiles; ++m)
        {
            queue = sycl_device.next_queue();

            // SYRK
            // ft_tiles[m * n_tiles + m] =
            //     hpx::dataflow(&syrk, queue, ft_tiles[m * n_tiles + k], ft_tiles[m * n_tiles + m], n_tile_size);
            ft_tiles[m * n_tiles + m] = hpx::dataflow(
                [&queue, n_tile_size](hpx::shared_future<double *> f_A, hpx::shared_future<double *> f_C)
                { return syrk(queue, f_A, f_C, n_tile_size); },
                ft_tiles[m * n_tiles + k], ft_tiles[m * n_tiles + m]
            );

            for (std::size_t n = k + 1; n < m; ++n)
            {
                queue = sycl_device.next_queue();

                // GEMM
                // ft_tiles[m * n_tiles + n] = hpx::dataflow(
                //     &gemm,
                //     queue,
                //     ft_tiles[m * n_tiles + k],
                //     ft_tiles[n * n_tiles + k],
                //     ft_tiles[m * n_tiles + n],
                //     n_tile_size,
                //     n_tile_size,
                //     n_tile_size,
                //     oneapi::math::transpose::nontrans,
                //     oneapi::math::transpose::trans);

                ft_tiles[m * n_tiles + n] = hpx::dataflow(
                    [&queue, n_tile_size]( 
                        hpx::shared_future<double *> f_A,
                        hpx::shared_future<double *> f_B,
                        hpx::shared_future<double *> f_C
                    )
                    { return gemm(queue, f_A, f_B, f_C, n_tile_size, n_tile_size, n_tile_size, oneapi::math::transpose::nontrans, oneapi::math::transpose::trans); },
                    ft_tiles[m * n_tiles + k], ft_tiles[n * n_tiles + k], ft_tiles[m * n_tiles + n]
                );
            }
        }
    }
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
        // ft_rhs[k] = hpx::dataflow(
        //     &trsv, 
        //     queue, 
        //     ft_tiles[k * n_tiles + k], 
        //     ft_rhs[k], 
        //     n_tile_size, 
        //     oneapi::math::transpose::nontrans
        // );
        ft_rhs[k] = hpx::dataflow(
            [&queue, n_tile_size](hpx::shared_future<double *> f_A, hpx::shared_future<double *> f_b)
            { return trsv(queue, f_A, f_b, n_tile_size, oneapi::math::transpose::nontrans); },
            ft_tiles[k * n_tiles + k], ft_rhs[k]
        );

        for (std::size_t m = k + 1; m < n_tiles; ++m)
        {
            queue = sycl_device.next_queue();

            // GEMV: b = b - A * a
            // ft_rhs[m] = hpx::dataflow(
            //     &gemv,
            //     queue,
            //     ft_tiles[m * n_tiles + k],
            //     ft_rhs[k],
            //     ft_rhs[m],
            //     n_tile_size,
            //     n_tile_size,
            //     -1,
            //     oneapi::math::transpose::nontrans
            // );
            ft_rhs[m] = hpx::dataflow(
                [&queue, n_tile_size](
                    hpx::shared_future<double *> f_A,
                    hpx::shared_future<double *> f_x,
                    hpx::shared_future<double *> f_y
                )
                { return gemv(queue, f_A, f_x, f_y, n_tile_size, n_tile_size, -1.0, oneapi::math::transpose::nontrans) },
                ft_tiles[m * n_tiles + k], ft_rhs[k], ft_rhs[m]
            );
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

    for (int k = n_tiles - 1; k >= 0; k--)
    {
        queue = sycl_device.next_queue();

        // TRSM: Solve L^T * x = a
        // ft_rhs[k] = hpx::dataflow(
        //     &trsv, 
        //     queue, 
        //     ft_tiles[k * n_tiles + k], 
        //     ft_rhs[k], 
        //     n_tile_size, 
        //     oneapi::math::transpose::trans
        // );
        ft_rhs[k] = hpx::dataflow(
            [&queue, n_tile_size](hpx::shared_future<double *> f_A, hpx::shared_future<double *> f_b)
            { return trsv(queue, f_A, f_b, n_tile_size, oneapi::math::transpose::trans); },
            ft_tiles[k * n_tiles + k], ft_rhs[k]
        );


        for (int m = k - 1; m >= 0; m--)
        {
            queue = sycl_device.next_queue();

            // GEMV: b = b - A^T * a
            // ft_rhs[m] = hpx::dataflow(
            //     &gemv,
            //     queue,
            //     ft_tiles[k * n_tiles + m],
            //     ft_rhs[k],
            //     ft_rhs[m],
            //     n_tile_size,
            //     n_tile_size,
            //     -1,
            //     oneapi::math::transpose::trans
            // );
            ft_rhs[m] = hpx::dataflow(
                [&queue, n_tile_size](
                    hpx::shared_future<double *> f_A,
                    hpx::shared_future<double *> f_x,
                    hpx::shared_future<double *> f_y
                )
                { return gemv(queue, f_A, f_x, f_y, n_tile_size, n_tile_size, -1.0, oneapi::math::transpose::trans); },
                ft_tiles[k * n_tiles + m], ft_rhs[k], ft_rhs[m]
            );
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
            // ft_rhs[k * m_tiles + c] = hpx::dataflow(
            //     &trsm,
            //     queue,
            //     ft_tiles[k * n_tiles + k],
            //     ft_rhs[k * m_tiles + c],
            //     n_tile_size,
            //     m_tile_size,
            //     oneapi::math::transpose::nontrans,
            //     oneapi::math::side::left
            // );
            ft_rhs[k * m_tiles + c] = hpx::dataflow(
                [&queue, n_tile_size, m_tile_size](hpx::shared_future<double *> f_A, hpx::shared_future<double *> f_B)
                { return trsm(queue, f_A, f_B, n_tile_size, m_tile_size, oneapi::math::transpose::nontrans, oneapi::math::side::left); },
                ft_tiles[k * n_tiles + k], ft_rhs[k * m_tiles + c]
            );


            for (std::size_t m = k + 1; m < n_tiles; ++m)
            {
                queue = sycl_device.next_queue();

                // GEMM: C = C - A * B
                // ft_rhs[m * m_tiles + c] = hpx::dataflow(
                //     &gemm,
                //     queue,
                //     ft_tiles[m * n_tiles + k],
                //     ft_rhs[k * m_tiles + c],
                //     ft_rhs[m * m_tiles + c],
                //     n_tile_size,
                //     m_tile_size,
                //     n_tile_size,
                //     oneapi::math::transpose::nontrans,
                //     oneapi::math::transpose::nontrans
                // );
                ft_rhs[m * m_tiles + c] = hpx::dataflow(
                    [&queue, n_tile_size, m_tile_size](
                        hpx::shared_future<double *> f_A,
                        hpx::shared_future<double *> f_B,
                        hpx::shared_future<double *> f_C
                    )
                    { return gemm(queue, f_A, f_B, f_C, n_tile_size, m_tile_size, n_tile_size, oneapi::math::transpose::nontrans, oneapi::math::transpose::nontrans); },
                    ft_tiles[m * n_tiles + k], ft_rhs[k * m_tiles + c], ft_rhs[m * m_tiles + c]
                );
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
            // ft_rhs[k * m_tiles + c] = hpx::dataflow(
            //     &trsm,
            //     queue,
            //     ft_tiles[k * n_tiles + k],
            //     ft_rhs[k * m_tiles + c],
            //     n_tile_size,
            //     m_tile_size,
            //     oneapi::math::transpose::trans,
            //     oneapi::math::side::left
            // );
            ft_rhs[k * m_tiles + c] = hpx::dataflow(
                [&queue, n_tile_size, m_tile_size](hpx::shared_future<double *> f_A, hpx::shared_future<double *> f_B)
                { return trsm(queue, f_A, f_B, n_tile_size, m_tile_size, oneapi::math::transpose::trans, oneapi::math::side::left); },
                ft_tiles[k * n_tiles + k], ft_rhs[k * m_tiles + c]
            );


            for (std::size_t m = 0; m < k; ++m)
            {
                queue = sycl_device.next_queue();

                // GEMM: C = C - A^T * B
                // ft_rhs[m * m_tiles + c] = hpx::dataflow(
                //     &gemm,
                //     queue,
                //     ft_tiles[k * n_tiles + m],
                //     ft_rhs[k * m_tiles + c],
                //     ft_rhs[m * m_tiles + c],
                //     n_tile_size,
                //     m_tile_size,
                //     n_tile_size,
                //     oneapi::math::transpose::trans,
                //     oneapi::math::transpose::nontrans
                // );
                ft_rhs[m * m_tiles + c] = hpx::dataflow(
                    [&queue, n_tile_size, m_tile_size](
                        hpx::shared_future<double *> f_A,
                        hpx::shared_future<double *> f_B,
                        hpx::shared_future<double *> f_C
                    )
                    { return gemm(queue, f_A, f_B, f_C, n_tile_size, m_tile_size, n_tile_size, oneapi::math::transpose::trans, oneapi::math::transpose::nontrans); },
                    ft_tiles[k * n_tiles + m], ft_rhs[k * m_tiles + c], ft_rhs[m * m_tiles + c]
                );
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

            // ft_rhs[k] = hpx::dataflow(
            //     &gemv,
            //     queue,
            //     ft_tiles[k * n_tiles + m],
            //     ft_vector[m],
            //     ft_rhs[k],
            //     N_row,
            //     N_col,
            //     1,
            //     oneapi::math::transpose::nontrans
            // );
            ft_rhs[k] = hpx::dataflow(
                [&queue, N_row, N_col](
                    hpx::shared_future<double *> f_A,
                    hpx::shared_future<double *> f_x,
                    hpx::shared_future<double *> f_y
                )
                { return gemv(queue, f_A, f_x, f_y, N_row, N_col, 1, oneapi::math::transpose::nontrans); },
                ft_tiles[k * n_tiles + m], ft_vector[m], ft_rhs[k]
            );
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
            // ft_inter_tiles[i] = hpx::dataflow(
            //     &dot_diag_syrk,
            //     queue,
            //     ft_tCC_tiles[n * m_tiles + i],
            //     ft_inter_tiles[i],
            //     n_tile_size,
            //     m_tile_size
            // );
            ft_inter_tiles[i] = hpx::dataflow(
                [&queue, n_tile_size, m_tile_size](hpx::shared_future<double *> f_A, hpx::shared_future<double *> f_r)
                { return dot_diag_syrk(queue, f_A, f_r, n_tile_size, m_tile_size); },
                ft_tCC_tiles[n * m_tiles + i], ft_inter_tiles[i]
            );
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

            // ft_alpha[i] = hpx::dataflow(
            //     &gemv,
            //     queue,
            //     ft_invK[i * n_tiles + j],
            //     ft_y[j],
            //     ft_alpha[i],
            //     n_tile_size,
            //     n_tile_size,
            //     1,
            //     oneapi::math::transpose::nontrans
            // );
            ft_alpha[i] = hpx::dataflow(
                [&queue, n_tile_size](
                    hpx::shared_future<double *> f_A,
                    hpx::shared_future<double *> f_x,
                    hpx::shared_future<double *> f_y
                )
                { return gemv(queue, f_A, f_x, f_y, n_tile_size, n_tile_size, 1, oneapi::math::transpose::nontrans); },
                ft_invK[i * n_tiles + j], ft_y[j], ft_alpha[i]
            );
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
        // loss_tiled[k] =
        //     hpx::dataflow(&compute_loss, ft_tiles[k * n_tiles + k], ft_alpha[k], ft_y[k], n_tile_size, std::ref(sycl_device));
        loss_tiled[k] = hpx::dataflow(
            [sycl_device, n_tile_size, n_tiles](
                const std::vector<double> &training_input,
                const std::vector<double> &training_output,
                const gprat_hyper::SEKParams &sek_params
            )
            { return compute_loss(training_input, training_output, sek_params, n_tile_size, std::ref(sycl_device)) },
            ft_tiles[k * n_tiles + k], ft_alpha[k], ft_y[k]
        );
    }

    // return hpx::dataflow(&add_losses, loss_tiled, n_tile_size, n_tiles);
    return hpx::dataflow(
        [n_tile_size, n_tiles](const std::vector<hpx::shared_future<double>> &losses)
        { return add_losses(losses, n_tile_size, n_tiles); },
        loss_tiled
    );
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
                // ft_priorK[c * m_tiles + k] = hpx::dataflow(
                //     &gemm,
                //     queue,
                //     ft_tCC_tiles[m * m_tiles + c],
                //     ft_tCC_tiles[m * m_tiles + k],
                //     ft_priorK[c * m_tiles + k],
                //     n_tile_size,
                //     m_tile_size,
                //     m_tile_size,
                //     oneapi::math::transpose::trans,
                //     oneapi::math::transpose::nontrans
                // );
                ft_priorK[c * m_tiles + k] = hpx::dataflow(
                    [&queue, n_tile_size, m_tile_size](
                        hpx::shared_future<double *> f_A,
                        hpx::shared_future<double *> f_B,
                        hpx::shared_future<double *> f_C
                    )
                    { return gemm(queue, f_A, f_B, f_C, n_tile_size, m_tile_size, m_tile_size, oneapi::math::transpose::trans, oneapi::math::transpose::nontrans); },
                    ft_tCC_tiles[m * m_tiles + c], ft_tCC_tiles[m * m_tiles + k], ft_priorK[c * m_tiles + k]
                );
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
        // ft_vector[i] = hpx::dataflow(&diag_posterior, ft_priorK[i], ft_inter[i], m_tile_size, std::ref(sycl_device));
        ft_vector[i] = hpx::dataflow(
            [sycl_device, m_tile_size](const hpx::shared_future<double *> A, const hpx::shared_future<double *> B)
            { return diag_posterior(A, B, m_tile_size, std::ref(sycl_device)); },
            ft_priorK[i], ft_inter[i]
        );
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
        // ft_vector[i] = hpx::dataflow(&diag_tile, ft_priorK[i * m_tiles + i], m_tile_size, std::ref(sycl_device));
        ft_vector[i] = hpx::dataflow(
            [m_tile_size](const hpx::shared_future<double *> A)
            { return diag_tile(A, m_tile_size, std::ref(sycl_device)); },
            ft_priorK[i * m_tiles + i]
        );
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

            // ft_tiles[i * n_tiles + j] =
            //     hpx::dataflow(&ger, queue, ft_tiles[i * n_tiles + j], ft_v1[i], ft_v2[j], n_tile_size);
            ft_tiles[i * n_tiles + j] = hpx::dataflow(
                [&queue, n_tile_size](
                    hpx::shared_future<double *> f_A,
                    hpx::shared_future<double *> f_x,
                    hpx::shared_future<double *> f_y
                )
                { return ger(queue, f_A, f_x, f_y); },
                ft_tiles[i * n_tiles + j], ft_v1[i], ft_v2[j]
            );

        }
    }
}

static double update_hyperparameter(
    const std::vector<hpx::shared_future<double *>> &ft_invK,
    const std::vector<hpx::shared_future<double *>> &ft_gradparam,
    const std::vector<hpx::shared_future<double *>> &ft_alpha,
    double &hyperparameter,  // lengthscale or vertical-lengthscale
    gprat_hyper::SEKParams sek_params,
    gprat_hyper::AdamParams adam_params,
    const std::size_t n_tile_size,
    const std::size_t n_tiles,
    std::vector<hpx::shared_future<double>> &m_T,
    std::vector<hpx::shared_future<double>> &v_T,
    const std::vector<hpx::shared_future<double>> &beta1_T,
    const std::vector<hpx::shared_future<double>> &beta2_T,
    int iter,
    int param_idx,  // 0 for lengthscale, 1 for vertical-lengthscale
    gprat::SYCL_DEVICE &sycl_device)
{
    throw std::logic_error("Function not implemented for GPU");
    // return 0;
}

double update_lengthscale(
    const std::vector<hpx::shared_future<double *>> &ft_invK,
    const std::vector<hpx::shared_future<double *>> &ft_gradparam,
    const std::vector<hpx::shared_future<double *>> &ft_alpha,
    gprat_hyper::SEKParams sek_params,
    gprat_hyper::AdamParams adam_params,
    const std::size_t n_tile_size,
    const std::size_t n_tiles,
    std::vector<hpx::shared_future<double>> &m_T,
    std::vector<hpx::shared_future<double>> &v_T,
    const std::vector<hpx::shared_future<double>> &beta1_T,
    const std::vector<hpx::shared_future<double>> &beta2_T,
    int iter,
    gprat::SYCL_DEVICE &sycl_device)
{
    return update_hyperparameter(
        ft_invK,
        ft_gradparam,
        ft_alpha,
        sek_params.lengthscale,
        sek_params,
        adam_params,
        n_tile_size,
        n_tiles,
        m_T,
        v_T,
        beta1_T,
        beta2_T,
        iter,
        0,
        sycl_device
    );
}

double update_vertical_lengthscale(
    const std::vector<hpx::shared_future<double *>> &ft_invK,
    const std::vector<hpx::shared_future<double *>> &ft_gradparam,
    const std::vector<hpx::shared_future<double *>> &ft_alpha,
    gprat_hyper::SEKParams sek_params,
    gprat_hyper::AdamParams adam_params,
    const std::size_t n_tile_size,
    const std::size_t n_tiles,
    std::vector<hpx::shared_future<double>> &m_T,
    std::vector<hpx::shared_future<double>> &v_T,
    const std::vector<hpx::shared_future<double>> &beta1_T,
    const std::vector<hpx::shared_future<double>> &beta2_T,
    int iter,
    gprat::SYCL_DEVICE &sycl_device
)
{
    return update_hyperparameter(
        ft_invK,
        ft_gradparam,
        ft_alpha,
        sek_params.vertical_lengthscale,
        sek_params,
        adam_params,
        n_tile_size,
        n_tiles,
        m_T,
        v_T,
        beta1_T,
        beta2_T,
        iter,
        1,
        sycl_device
    );
}

double update_noise_variance(
    const std::vector<hpx::shared_future<double *>> &ft_invK,
    const std::vector<hpx::shared_future<double *>> &ft_alpha,
    gprat_hyper::SEKParams sek_params,
    gprat_hyper::AdamParams adam_params,
    const std::size_t n_tile_size,
    const std::size_t n_tiles,
    std::vector<hpx::shared_future<double>> &m_T,
    std::vector<hpx::shared_future<double>> &v_T,
    const std::vector<hpx::shared_future<double>> &beta1_T,
    const std::vector<hpx::shared_future<double>> &beta2_T,
    int iter,
    gprat::SYCL_DEVICE &sycl_device
)
{
    throw std::logic_error("Function not implemented for GPU");
    // return 0;
}

}  // end of namespace sycl_backend
