#include "gpu/sycl/sycl_gp_optimizer.hpp"
#include "gpu/sycl/adapter_onemath.hpp"
#include "gpu/sycl/sycl_kernels.hpp"
#include "gpu/sycl/sycl_utils.hpp"

namespace gprat::sycl_backend
{

// no SYCL functionality //////////////////////////////////////////////////////////////////////////////////////////////

// double to_constrained(const double parameter, bool noise)
// {
//     if (noise)
//     {
//         return log(1.0 + exp(parameter)) + 1e-6;
//     }
//     else
//     {
//         return log(1.0 + exp(parameter));
//     }
// }

// double to_unconstrained(const double parameter, bool noise)
// {
//     if (noise)
//     {
//         return log(exp(parameter - 1e-6) - 1.0);
//     }
//     else
//     {
//         return log(exp(parameter) - 1.0);
//     }
// }

// double compute_sigmoid(const double parameter) { return 1.0 / (1.0 + exp(-parameter)); }

// double compute_covariance_distance(
//     std::size_t i_global,
//     std::size_t j_global,
//     std::size_t n_regressors,
//     gprat_hyper::SEKParams sek_params,
//     const std::vector<double> &i_input,
//     const std::vector<double> &j_input
// )
// {
//     // C(z_i,z_j) = vertical_lengthscale * exp(-0.5*lengthscale*(z_i-z_j)^2)
//     double distance = 0.0;
//     double z_ik_minus_z_jk;

//     for (std::size_t k = 0; k < n_regressors; ++k)
//     {
//         z_ik_minus_z_jk = i_input[i_global + k] - j_input[j_global + k];
//         distance += z_ik_minus_z_jk * z_ik_minus_z_jk;
//     }

//     return -1.0 / (2.0 * pow(sek_params.lengthscale, 2.0)) * distance;
// }

// std::vector<double> gen_tile_distance(
//     std::size_t row,
//     std::size_t col,
//     std::size_t N,
//     std::size_t n_regressors,
//     gprat_hyper::SEKParams sek_params,
//     const std::vector<double> &input
// )
// {
//     std::size_t i_global, j_global;
//     double cov_dist;
//     // Initialize tile
//     std::vector<double> tile;
//     tile.resize(N * N);
//     for (std::size_t i = 0; i < N; i++)
//     {
//         i_global = N * row + i;
//         for (std::size_t j = 0; j < N; j++)
//         {
//             j_global = N * col + j;
//             // compute covariance function
//             cov_dist = compute_covariance_distance(i_global, j_global, n_regressors, sek_params, input, input);
//             tile[i * N + j] = cov_dist;
//         }
//     }
//     return tile;
// }

// std::vector<double> gen_tile_covariance_with_distance(
//     std::size_t row,
//     std::size_t col,
//     std::size_t N,
//     // std::size_t n_regressors,
//     gprat_hyper::SEKParams sek_params,
//     const std::vector<double> &cov_dists
// )
// {
//     std::size_t i_global, j_global;
//     double covariance;
//     // Initialize tile
//     std::vector<double> tile;
//     tile.resize(N * N);
//     for (std::size_t i = 0; i < N; i++)
//     {
//         i_global = N * row + i;
//         for (std::size_t j = 0; j < N; j++)
//         {
//             j_global = N * col + j;
//             // compute covariance function
//             covariance = sek_params.vertical_lengthscale * exp(cov_dists[i * N + j]);
//             if (i_global == j_global)
//             {
//                 // noise variance on diagonal
//                 covariance += sek_params.noise_variance;
//             }
//             tile[i * N + j] = covariance;
//         }
//     }
//     return tile;
// }

// std::vector<double>
// gen_tile_grad_v(
//     std::size_t N,
//     gprat_hyper::SEKParams sek_params,
//     const std::vector<double> &cov_dists
// )
// {
//     // Initialize tile
//     std::vector<double> tile;
//     tile.resize(N * N);
//     double hyperparam_der = compute_sigmoid(to_unconstrained(sek_params.vertical_lengthscale, false));
//     for (std::size_t i = 0; i < N; i++)
//     {
//         for (std::size_t j = 0; j < N; j++)
//         {
//             // compute covariance function
//             tile[i * N + j] = exp(cov_dists[i * N + j]) * hyperparam_der;
//         }
//     }
//     return tile;
// }

// std::vector<double>
// gen_tile_grad_l(
//     std::size_t N,
//     gprat_hyper::SEKParams sek_params,
//     const std::vector<double> &cov_dists
// )
// {
//     // Initialize tile
//     std::vector<double> tile;
//     tile.resize(N * N);
//     double hyperparam_der = compute_sigmoid(to_unconstrained(sek_params.lengthscale, false));
//     for (std::size_t i = 0; i < N; i++)
//     {
//         for (std::size_t j = 0; j < N; j++)
//         {
//             // compute covariance function
//             tile[i * N + j] = -2.0 * (sek_params.vertical_lengthscale / sek_params.lengthscale) * cov_dists[i * N + j]
//                               * exp(cov_dists[i * N + j]) * hyperparam_der;
//         }
//     }
//     return tile;
// }

// std::vector<double> gen_tile_grad_v_trans(std::size_t N, const std::vector<double> &grad_l_tile)
// {
//     std::vector<double> transposed;
//     transposed.resize(N * N);
//     for (std::size_t i = 0; i < N; i++)
//     {
//         for (std::size_t j = 0; j < N; j++)
//         {
//             // Mapping (i, j) in the original matrix to (j, i) in the transposed
//             // matrix
//             transposed[j * N + i] = grad_l_tile[i * N + j];
//         }
//     }
//     return transposed;
// }

// // Contains SYCL functionality

// gen_tile_grad_l_trans //////////////////////////////////////////////////////////////////////////////////////////////

hpx::shared_future<double *>
gen_tile_grad_l_trans(std::size_t N, const hpx::shared_future<double *> f_grad_l_tile, gprat::SYCL_DEVICE &sycl_device)
{
    try
    {
        sycl::queue queue = sycl_device.next_queue();

        double *transposed = sycl::malloc_device<double>(N * N, queue);
        double *d_grad_l_tile = f_grad_l_tile.get();

        sycl::range<2> global_range(
            ((N + WORK_GROUP_SIZE - 1) / WORK_GROUP_SIZE) * WORK_GROUP_SIZE,
            ((N + WORK_GROUP_SIZE - 1) / WORK_GROUP_SIZE) * WORK_GROUP_SIZE
        );
        sycl::range<2> local_range(WORK_GROUP_SIZE, WORK_GROUP_SIZE);

        auto event = queue.submit([&](sycl::handler &cgh) {
            cgh.parallel_for(
                sycl::nd_range<2>(global_range, local_range),
                [=](sycl::nd_item<2> item) {
                    std::size_t row = item.get_global_id(0);
                    std::size_t col = item.get_global_id(1);

                    if (row < N && col < N) {
                        transposed[row * N + col] = d_grad_l_tile[col * N + row];
                    }
                }
            );
        });

        event.wait();
        return hpx::make_ready_future(transposed);
    }
    catch (const sycl::exception &e)
    {
        std::cout << "SYCL exception: " << e.what() << "\n";
        return hpx::make_ready_future(static_cast<double *>(nullptr));
    }
}

// gen_beta_T /////////////////////////////////////////////////////////////////////////////////////////////////////////

// double gen_beta_T(int t, double beta) { return std::pow(beta, t); }

// compute_loss ///////////////////////////////////////////////////////////////////////////////////////////////////////

double compute_loss(
    const hpx::shared_future<double *> &K_diag_tile,
    const hpx::shared_future<double *> &alpha_tile,
    const hpx::shared_future<double *> &y_tile,
    std::size_t N,
    gprat::SYCL_DEVICE &sycl_device
)
{
    sycl::queue queue = sycl_device.next_queue();
    // l = y^T * alpha + \sum_i^N log(L_ii^2)
    double l;
    // Compute y^T * alpha
    l = *(dot(queue, y_tile.get(), alpha_tile.get(), N));
    // Compute \sum_i^N log(L_ii^2)
    for (std::size_t i = 0; i < N; i++)
    {
        double diag_value = K_diag_tile.get()[i * N + i];
        l += std::log(diag_value * diag_value);
    }
    return l;
}

// add_losses /////////////////////////////////////////////////////////////////////////////////////////////////////////

hpx::shared_future<double>
add_losses(const std::vector<hpx::shared_future<double>> &losses, std::size_t n_tile_size, std::size_t n_tiles)
{
    // Add the squared difference to the error
    double l = 0.0;
    for (std::size_t i = 0; i < n_tiles; i++)
    {
        l += losses[i].get();
    }
    l += static_cast<double>(n_tile_size) * static_cast<double>(n_tiles) * log(2.0 * M_PI);

    return hpx::make_ready_future(0.5 * l / static_cast<double>((n_tile_size * n_tiles)));
}

// double 
// compute_gradient(const double &grad_l, const double &grad_r, std::size_t N, std::size_t n_tiles)
// {
//     double grad = 0.0;
//     grad = 1.0 / (2.0 * static_cast<double>(N) * static_cast<double>(n_tiles)) * (grad_l - grad_r);

//     return std::move(grad);
// }

// double 
// compute_gradient_noise(
//     const std::vector<std::vector<double>> &ft_tiles, 
//     double noise_variance, 
//     std::size_t N, 
//     std::size_t n_tiles
// )
// {
//     // Initialize tile
//     double trace = 0.0;
//     double hyperparam_der = compute_sigmoid(to_unconstrained(noise_variance, true));
//     for (std::size_t d = 0; d < n_tiles; d++)
//     {
//         auto tile = ft_tiles[d * n_tiles + d];
//         for (std::size_t i = 0; i < N; ++i)
//         {
//             trace += (tile[i * N + i] * hyperparam_der);
//         }
//     }
//     trace = 1.0 / (2.0 * static_cast<double>(N) * static_cast<double>(n_tiles)) * trace;
//     return std::move(trace);
// }

// double update_first_moment(const double &gradient, double m_T, const double &beta_1)
// {
//     return beta_1 * m_T + (1.0 - beta_1) * gradient;
// }

// double update_second_moment(const double &gradient, double v_T, const double &beta_2)
// {
//     return beta_2 * v_T + (1.0 - beta_2) * gradient * gradient;
// }

// hpx::shared_future<double>
// update_param(
// const double unconstrained_hyperparam,
// gprat_hyper::SEKParams sek_params,
// gprat_hyper::AdamParams adam_params,
// double m_T,
// double v_T,
// const std::vector<double> beta1_T,
// const std::vector<double> beta2_T,
// int iter
// )
// {
//     double alpha_T = sek_params.noise_variance * sqrt(1.0 - beta2_T[static_cast<size_t>(iter)]) / (1.0 - beta1_T[static_cast<size_t>(iter)]);
//     return hpx::make_ready_future(unconstrained_hyperparam - alpha_T * m_T / (sqrt(v_T) + adam_params.epsilon));
// }

// std::vector<double> 
// gen_tile_identity(std::size_t row, std::size_t col, std::size_t N)
// {
//     std::size_t i_global, j_global;
//     // Initialize tile
//     std::vector<double> tile;
//     tile.resize(N * N);
//     std::fill(tile.begin(), tile.end(), 0.0);
//     for (std::size_t i = 0; i < N; i++)
//     {
//         i_global = N * row + i;
//         for (std::size_t j = i; j <= i; j++)
//         {
//             j_global = N * col + j;
//             if (i_global == j_global)
//             {
//                 tile[i * N + j] = 1.0;
//             }
//         }
//     }
//     return tile;
// }

// std::vector<double> 
// gen_tile_zeros_diag(std::size_t N)
// {
//     // Initialize tile
//     std::vector<double> tile;
//     tile.resize(N);
//     std::fill(tile.begin(), tile.end(), 0.0);
//     return tile;
// }

// double 
// gen_moment()
// {
//     double z = 0.0;
//     return z;
// }

// double 
// sum_gradleft(const std::vector<double> &diagonal, double grad)
// {
//     grad += std::reduce(diagonal.begin(), diagonal.end());
//     return grad;
// }

// double
// sum_gradright(
//     // const std::vector<double> &inter_alpha, const std::vector<double> &alpha, double grad, std::size_t N
// )
// {
//     // grad += dot(inter_alpha, alpha, N);
//     // return grad;
//     return 0.0;
// }

// double 
// sum_noise_gradleft(
//     const std::vector<double> &ft_invK,
//     double grad,
//     gprat_hyper::SEKParams sek_params,
//     std::size_t N//,
//     // std::size_t n_tiles
// )
// {
//     double noise_der = compute_sigmoid(to_unconstrained(sek_params.noise_variance, true));
//     for (std::size_t i = 0; i < N; ++i)
//     {
//         grad += (ft_invK[i * N + i] * noise_der);
//     }
//     return std::move(grad);
// }

// double
// sum_noise_gradright(
//     //const std::vector<double> &alpha, double grad, gprat_hyper::SEKParams sek_params, std::size_t N
// )
// {
//     // double noise_der =
//     //     compute_sigmoid(to_unconstrained(sek_params.noise_variance, true));
//     // grad += (noise_der * dot(alpha, alpha, N));
//     // return grad;
//     return 0.0;
// }

}  // end of namespace sycl_backend
