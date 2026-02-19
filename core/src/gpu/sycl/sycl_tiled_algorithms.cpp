#include "gpu/sycl/sycl_tiled_algorithms.hpp"
#include "gpu/sycl/adapter_onemath.hpp"
#include "gpu/sycl/sycl_gp_optimizer.hpp"
#include "gpu/sycl/sycl_gp_uncertainty.hpp"

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
    double *result;

    for (std::size_t k = 0; k < n_tiles; ++k)
    {
        result = potrf(sycl_device.next_queue(), ft_tiles[k * n_tiles + k].get(), n_tile_size);
        ft_tiles[k * n_tiles + k] = hpx::make_ready_future(result);

        for (std::size_t m = k + 1; m < n_tiles; ++m)
        {
            result = trsm(
                sycl_device.next_queue(), 
                ft_tiles[k * n_tiles + k].get(), 
                ft_tiles[m * n_tiles + k].get(),
                n_tile_size, n_tile_size, 
                oneapi::math::transpose::trans, 
                oneapi::math::side::right
            );

            ft_tiles[m * n_tiles + k] = hpx::make_ready_future(result);
        }

        for (std::size_t m = k + 1; m < n_tiles; ++m)
        {
            result = syrk(
                sycl_device.next_queue(), 
                ft_tiles[m * n_tiles + k].get(), 
                ft_tiles[m * n_tiles + m].get(), 
                n_tile_size
            );

            ft_tiles[m * n_tiles + m] = hpx::make_ready_future(result);

            for (std::size_t n = k + 1; n < m; ++n)
            {
                result = gemm(
                    sycl_device.next_queue(), 
                    ft_tiles[m * n_tiles + k].get(), 
                    ft_tiles[n * n_tiles + k].get(), 
                    ft_tiles[m * n_tiles + n].get(), 
                    n_tile_size, n_tile_size, n_tile_size, 
                    oneapi::math::transpose::nontrans, 
                    oneapi::math::transpose::trans
                );

                ft_tiles[m * n_tiles + n] = hpx::make_ready_future(result);
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
    double *result;
    double *result_gemv;
    double *test1;
    double *test2;

    for (std::size_t k = 0; k < n_tiles; ++k)
    {
        // TRSM: Solve L * x = a
        // ft_rhs[k] = hpx::dataflow(
        //     hpx::unwrapping(&trsv), 
        //     f_trsm,
        //     ft_tiles[k * n_tiles + k], 
        //     ft_rhs[k], 
        //     n_tile_size, 
        //     oneapi::math::transpose::nontrans);

        result = trsv(
            sycl_device.next_queue(), 
            ft_tiles[k * n_tiles + k].get(), 
            ft_rhs[k].get(), 
            n_tile_size, 
            oneapi::math::transpose::nontrans
        );

        ft_rhs[k] = hpx::make_ready_future(result);

        auto gemv_queue = sycl_device.next_queue();

        // std::cout << "STOP! HAMMERTIME" << std::endl;
        // std::vector<double> test_vector(n_tile_size * n_tile_size, -1.0);
        // auto copy_process = gemv_queue.memcpy(test_vector.data(), result, test_vector.size() * sizeof(double));
        // copy_process.wait();

        // std::cout << "Test vector has length " << test_vector.size() << " and contents: \n [";
        // for (int i = 0; i < test_vector.size(); ++i)
        // {
        //     std::cout << test_vector[i] << " ";
        // }
        // std::cout << "]\n";

        // std::cout << "STOP! HAMMERTIME 2" << std::endl;
        // std::vector<double> ft_vector(n_tile_size * n_tile_size, -1.0);
        // copy_process = gemv_queue.memcpy(ft_vector.data(), ft_tiles[k * n_tiles + k].get(), ft_vector.size() * sizeof(double));
        // copy_process.wait();

        // std::cout << "FT vector has length " << ft_vector.size() << " and contents: \n [";
        // for (int i = 0; i < ft_vector.size(); ++i)
        // {
        //     std::cout << ft_vector[i] << " ";
        // }
        // std::cout << "]\n";

        // std::cout << "STOP! HAMMERTIME 3" << std::endl;
        // std::vector<double> rhs_vector(n_tile_size * n_tile_size, -1.0);
        // copy_process = gemv_queue.memcpy(rhs_vector.data(), ft_rhs[k].get(), rhs_vector.size() * sizeof(double));
        // copy_process.wait();

        // std::cout << "RHS vector has length " << rhs_vector.size() << " and contents: \n [";
        // for (int i = 0; i < rhs_vector.size(); ++i)
        // {
        //     std::cout << rhs_vector[i] << " ";
        // }
        // std::cout << "]\n";


        for (std::size_t m = k + 1; m < n_tiles; ++m)
        {
            // GEMV: b = b - A * a
            // ft_rhs[m] = hpx::dataflow(
            //     hpx::unwrapping(&gemv),
            //     f_gemv,
            //     ft_tiles[m * n_tiles + k],
            //     ft_rhs[k],
            //     ft_rhs[m],
            //     n_tile_size,
            //     n_tile_size,
            //     -1,
            //     oneapi::math::transpose::nontrans);

            test1 = ft_rhs[m].get();
            test2 = ft_tiles[m * n_tiles + k].get();

            // std::vector<double> test_vector(n_tile_size * n_tile_size, -1.0);
            // auto copy_process = gemv_queue.memcpy(test2, test_vector.data(), test_vector.size() * sizeof(double));
            // copy_process.wait();

            // double *vector1_imitate = sycl::malloc_device<gprat::sycl_backend::real_t>(n_tile_size, gemv_queue);

            // std::vector<double> test1_vector(n_tile_size, -2.0);
            // copy_process = gemv_queue.memcpy(vector1_imitate, test1_vector.data(), test1_vector.size() * sizeof(double));
            // copy_process.wait();

            // double *vector2_imitate = sycl::malloc_device<gprat::sycl_backend::real_t>(n_tile_size, gemv_queue);

            // std::vector<double> test2_vector(n_tile_size, -3.0);
            // copy_process = gemv_queue.memcpy(vector2_imitate, test2_vector.data(), test2_vector.size() * sizeof(double));
            // copy_process.wait();

            result_gemv = gemv(
                gemv_queue, 
                test2, 
                result, 
                test1, 
                n_tile_size, n_tile_size, 
                -1, 
                oneapi::math::transpose::nontrans
            );

            ft_rhs[m] = hpx::make_ready_future(result_gemv);
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
    double *result;
    // NOTE: The loops traverse backwards. Its last comparisons require the
    // usage negative numbers. Therefore they use signed int instead of the
    // unsigned std::size_t.

    for (int k = static_cast<int>(n_tiles) - 1; k >= 0; k--)
    {
        // // TRSM: Solve L^T * x = a
        // ft_rhs[static_cast<std::size_t>(k)] = hpx::dataflow(
        //     hpx::unwrapping(&trsv),
        //     f_trsv,
        //     ft_tiles[k * n_tiles + k], 
        //     ft_rhs[static_cast<std::size_t>(k)], 
        //     n_tile_size, 
        //     oneapi::math::transpose::trans);

        result = trsv(
            sycl_device.next_queue(), 
            ft_tiles[k * n_tiles + k].get(), 
            ft_rhs[static_cast<std::size_t>(k)].get(), 
            n_tile_size, 
            oneapi::math::transpose::trans
        );

        ft_rhs[static_cast<std::size_t>(k)] = hpx::make_ready_future(result);

        for (int m = k - 1; m >= 0; m--)
        {
            // // GEMV: b = b - A^T * a
            // ft_rhs[static_cast<std::size_t>(m)] = hpx::dataflow(
            //     hpx::unwrapping(&gemv),
            //     f_gemv,
            //     ft_tiles[k * n_tiles + m],
            //     ft_rhs[static_cast<std::size_t>(k)],
            //     ft_rhs[static_cast<std::size_t>(m)],
            //     n_tile_size,
            //     n_tile_size,
            //     -1,
            //     oneapi::math::transpose::trans
            // );

            result = gemv(
                sycl_device.next_queue(), 
                ft_tiles[k * n_tiles + m].get(), 
                ft_rhs[static_cast<std::size_t>(k)].get(), 
                ft_rhs[static_cast<std::size_t>(m)].get(), 
                n_tile_size, n_tile_size, 
                -1, 
                oneapi::math::transpose::trans
            );

            ft_rhs[static_cast<std::size_t>(m)] = hpx::make_ready_future(result);
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
    double *result;
    double *result_gemm;

    for (std::size_t c = 0; c < m_tiles; ++c)
    {
        for (std::size_t k = 0; k < n_tiles; ++k)
        {
            // TRSM: solve L * X = A
            // ft_rhs[static_cast<std::size_t>(static_cast<std::size_t>(k * m_tiles + c))] = hpx::dataflow(
            //     hpx::unwrapping(&trsm),
            //     f_trsm,
            //     ft_tiles[static_cast<std::size_t>(k) * n_tiles + static_cast<std::size_t>(k)],
            //     ft_rhs[static_cast<std::size_t>(k * m_tiles + c)],
            //     n_tile_size,
            //     m_tile_size,
            //     oneapi::math::transpose::nontrans,
            //     hpx::make_ready_future(oneapi::math::side::left));

            result = trsm(
                sycl_device.next_queue(), 
                ft_tiles[static_cast<std::size_t>(k) * n_tiles + static_cast<std::size_t>(k)].get(), 
                ft_rhs[static_cast<std::size_t>(k * m_tiles + c)].get(), 
                n_tile_size, m_tile_size, 
                oneapi::math::transpose::nontrans, 
                oneapi::math::side::left
            );

            ft_rhs[static_cast<std::size_t>(k * m_tiles + c)] = hpx::make_ready_future(result);
    
            for (std::size_t m = k + 1; m < n_tiles; ++m)
            {
                // GEMM: C = C - A * B
                // ft_rhs[m * m_tiles + c] = hpx::dataflow(
                //     hpx::unwrapping(&gemm),
                //     f_gemm,
                //     ft_tiles[m * n_tiles + k],
                //     ft_rhs[static_cast<std::size_t>(k * m_tiles + c)],
                //     ft_rhs[m * m_tiles + c],
                //     n_tile_size,
                //     m_tile_size,
                //     n_tile_size,
                //     oneapi::math::transpose::nontrans,
                //     oneapi::math::transpose::nontrans
                // );

                // std::cout << "Forward solve tiled matrix: c = " << c << ", k = " << k << ", m = " << m << std::endl;
                result_gemm = gemm(
                    sycl_device.next_queue(), 
                    ft_tiles[m * n_tiles + k].get(), 
                    ft_rhs[static_cast<std::size_t>(k * m_tiles + c)].get(), 
                    ft_rhs[m * m_tiles + c].get(), 
                    n_tile_size, m_tile_size, n_tile_size, 
                    oneapi::math::transpose::nontrans, 
                    oneapi::math::transpose::nontrans
                );

                ft_rhs[m * m_tiles + c] = hpx::make_ready_future(result_gemm);
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
    double *result;

    for (std::size_t c = 0; c < m_tiles; ++c)
    {
        for (std::size_t k = 0; k < n_tiles; ++k)
        {    
            // TRSM: solve L^T * X = A
            // ft_rhs[static_cast<std::size_t>(k * m_tiles + c)] = hpx::dataflow(
            //     hpx::unwrapping(&trsm),
            //     f_trsm,
            //     ft_tiles[static_cast<std::size_t>(k) * n_tiles + static_cast<std::size_t>(k)],
            //     ft_rhs[static_cast<std::size_t>(k * m_tiles + c)],
            //     n_tile_size,
            //     m_tile_size,
            //     oneapi::math::transpose::trans,
            //     hpx::make_ready_future(oneapi::math::side::left)
            // );

            result = trsm(
                sycl_device.next_queue(), 
                ft_tiles[static_cast<std::size_t>(k) * n_tiles + static_cast<std::size_t>(k)].get(), 
                ft_rhs[static_cast<std::size_t>(k * m_tiles + c)].get(), 
                n_tile_size, m_tile_size, 
                oneapi::math::transpose::trans, 
                oneapi::math::side::left
            ); 

            ft_rhs[static_cast<std::size_t>(k * m_tiles + c)] = hpx::make_ready_future(result);
    
            for (std::size_t m = 0; m < k; ++m)
            {
                // GEMM: C = C - A^T * B
                // ft_rhs[m * m_tiles + c] = hpx::dataflow(
                //     hpx::unwrapping(&gemm),
                //     f_gemm,
                //     ft_tiles[k * n_tiles + m],
                //     ft_rhs[static_cast<std::size_t>(k * m_tiles + c)],
                //     ft_rhs[m * m_tiles + c],
                //     n_tile_size,
                //     m_tile_size,
                //     n_tile_size,
                //     oneapi::math::transpose::trans,
                //     oneapi::math::transpose::nontrans
                // );

                result = gemm(
                    sycl_device.next_queue(), 
                    ft_tiles[k * n_tiles + m].get(), 
                    ft_rhs[static_cast<std::size_t>(k * m_tiles + c)].get(), 
                    ft_rhs[m * m_tiles + c].get(), 
                    n_tile_size, m_tile_size, n_tile_size, 
                    oneapi::math::transpose::trans, 
                    oneapi::math::transpose::nontrans
                );

                ft_rhs[m * m_tiles + c] = hpx::make_ready_future(result);
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
    double *result;

    for (std::size_t k = 0; k < m_tiles; ++k)
    {
        for (std::size_t m = 0; m < n_tiles; ++m)
        {
            // ft_rhs[k] = hpx::dataflow(
            //     hpx::unwrapping(&gemv),
            //     f_gemv,
            //     ft_tiles[k * n_tiles + m],
            //     ft_vector[m],
            //     ft_rhs[k],
            //     N_row,
            //     N_col,
            //     1,
            //     oneapi::math::transpose::nontrans
            // );

            result = gemv(
                sycl_device.next_queue(), 
                ft_tiles[k * n_tiles + m].get(), 
                ft_vector[m].get(), 
                ft_rhs[k].get(), 
                N_row, N_col, 
                1, 
                oneapi::math::transpose::nontrans
            );

            ft_rhs[k] = hpx::make_ready_future(result);
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
    double *result;

    for (std::size_t i = 0; i < m_tiles; ++i)
    {
        for (std::size_t n = 0; n < n_tiles; ++n)
        {
            hpx::shared_future<sycl::queue> f_dot_diag_syrk = hpx::make_ready_future(sycl_device.next_queue());

            // Compute inner product to obtain diagonal elements of
            // (K_MxN * (K^-1_NxN * K_NxM))
            // ft_inter_tiles[i] = hpx::dataflow(
            //     hpx::unwrapping(&dot_diag_syrk),
            //     f_dot_diag_syrk,
            //     ft_tCC_tiles[n * m_tiles + i],
            //     ft_inter_tiles[i],
            //     n_tile_size,
            //     m_tile_size
            // );

            result = dot_diag_syrk(
                sycl_device.next_queue(), 
                ft_tCC_tiles[n * m_tiles + i].get(), 
                ft_inter_tiles[i].get(), 
                n_tile_size, m_tile_size
            );

            ft_inter_tiles[i] = hpx::make_ready_future(result);
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
    double *result;

    for (std::size_t i = 0; i < n_tiles; ++i)
    {
        for (std::size_t j = 0; j < n_tiles; ++j)
        {
            // ft_alpha[i] = hpx::dataflow(
            //     hpx::unwrapping(&gemv),
            //     f_gemv,
            //     ft_invK[i * n_tiles + j],
            //     ft_y[j],
            //     ft_alpha[i],
            //     n_tile_size,
            //     n_tile_size,
            //     1,
            //     oneapi::math::transpose::nontrans
            // );

            result = gemv(
                sycl_device.next_queue(), 
                ft_invK[i * n_tiles + j].get(), 
                ft_y[j].get(), 
                ft_alpha[i].get(), 
                n_tile_size, n_tile_size, 
                1, 
                oneapi::math::transpose::nontrans
            );

            ft_alpha[i] = hpx::make_ready_future(result);
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
    double *result;

    for (std::size_t c = 0; c < m_tiles; ++c)
    {
        for (std::size_t k = 0; k < m_tiles; ++k)
        {
            for (std::size_t m = 0; m < n_tiles; ++m)
            {
                // GEMM:  C = C - A^T * B
                // ft_priorK[c * m_tiles + k] = hpx::dataflow(
                //     hpx::unwrapping(&gemm),
                //     f_gemm,
                //     ft_tCC_tiles[m * m_tiles + c],
                //     ft_tCC_tiles[m * m_tiles + k],
                //     ft_priorK[c * m_tiles + k],
                //     n_tile_size,
                //     m_tile_size,
                //     m_tile_size,
                //     oneapi::math::transpose::trans,
                //     oneapi::math::transpose::nontrans
                // );

                result = gemm(
                    sycl_device.next_queue(), 
                    ft_tCC_tiles[m * m_tiles + c].get(), 
                    ft_tCC_tiles[m * m_tiles + k].get(), 
                    ft_priorK[c * m_tiles + k].get(), 
                    n_tile_size, m_tile_size, m_tile_size, 
                    oneapi::math::transpose::trans, 
                    oneapi::math::transpose::nontrans
                );

                ft_priorK[c * m_tiles + k] = hpx::make_ready_future(result);
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
        ft_vector[i] = hpx::dataflow(hpx::unwrapping(&diag_posterior), ft_priorK[i], ft_inter[i], m_tile_size);
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
        ft_vector[i] = hpx::dataflow(hpx::unwrapping(&diag_tile), ft_priorK[i * m_tiles + i], m_tile_size);
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
    double *result;

    for (std::size_t i = 0; i < n_tiles; ++i)
    {
        for (std::size_t j = 0; j < n_tiles; ++j)
        {
            // hpx::shared_future<sycl::queue> f_ger = hpx::make_ready_future(sycl_device.next_queue());

            // ft_tiles[i * n_tiles + j] =
            //     hpx::dataflow(hpx::unwrapping(&ger), f_ger, ft_tiles[i * n_tiles + j], ft_v1[i], ft_v2[j], n_tile_size);

            result = ger(
                sycl_device.next_queue(), 
                ft_tiles[i * n_tiles + j].get(), 
                ft_v1[i].get(), 
                ft_v2[j].get(), 
                n_tile_size
            );

            ft_tiles[i * n_tiles + j] = hpx::make_ready_future(result);
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
