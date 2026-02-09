#include "sycl/sycl_gp_functions.hpp"

#include "gp_kernels.hpp"
#include "sycl/sycl_utils.hpp"
#include "sycl/sycl_gp_algorithms.hpp"
#include "sycl/sycl_tiled_algorithms.hpp"

#include "target.hpp"

#include <hpx/algorithm.hpp>

namespace gprat::sycl_backend
{

// predict ////////////////////////////////////////////////////////////////////////////////////////////////////////////

std::vector<double>
predict(
    const std::vector<double> &h_training_input,
    const std::vector<double> &h_training_output,
    const std::vector<double> &h_test_input,
    const gprat_hyper::SEKParams &sek_params,
    int n_tiles,
    int n_tile_size,
    int m_tiles,
    int m_tile_size,
    int n_regressors,
    gprat::SYCL_DEVICE &sycl_device
)
{
    sycl_device.create();

    std::cout << "[sycl_gp_functions.cpp] [predict] : Creating device pointers \n";

    double *d_training_input = copy_to_device(h_training_input, sycl_device, 1);
    double *d_training_output = copy_to_device(h_training_output, sycl_device);
    double *d_test_input = copy_to_device(h_test_input, sycl_device);

    std::cout << "[sycl_gp_functions.cpp] [predict] : DONE creating device pointers \n";

    // // DEBUG

    // sycl::queue debug_queue = sycl_device.next_queue();

    // std::cout << "\033[33m:::::::: TEST WITHIN SYCL_GP_FUNCTIONS::PREDICT ::::::::\n";
    // std::cout << "Expected length: " << h_training_input.size() << "\n";

    // std::vector<real_t> test(h_training_input.size());
    // auto copy_process = debug_queue.memcpy(test.data(), d_training_input, h_training_input.size() * sizeof(real_t));
    // std::cout << "Actual length: " << test.size() << "\n";
    // std::cout << "Contents: \n [";

    // for (int i = 0; i < test.size(); ++i)
    // {
    //     std::cout << test[i] << " ";
    // }
    // std::cout << "]\033[0m\n\n" << std::endl;

    // // END OF DEBUG

    // std::cout << "[gprat::sycl_backend::predict] d_training_input (AKA d_input) = " << d_training_input << std::endl;

    std::cout << "[sycl_gp_functions.cpp] [predict] : Accessing assemble_tiled_covariance_matrix \n";

    auto d_tiles = assemble_tiled_covariance_matrix(
        d_training_input, 
        static_cast<std::size_t>(n_tiles), 
        static_cast<std::size_t>(n_tile_size),  
        static_cast<std::size_t>(n_regressors),   
        sek_params, 
        sycl_device
    );

    std::cout << "[sycl_gp_functions.cpp] [predict] : d_tiles has size " << d_tiles.size() << "\n";

    std::cout << "[sycl_gp_functions.cpp] [predict] : Accessing assemble_alpha_tiles \n";

    auto alpha_tiles = assemble_alpha_tiles(d_training_output, static_cast<std::size_t>(n_tiles), static_cast<std::size_t>(n_tile_size), sycl_device);

    std::cout << "[sycl_gp_functions.cpp] [predict] : Accessing assemble_cross_covariance_tiles \n";

    auto cross_covariance_tiles = assemble_cross_covariance_tiles(
        d_test_input, 
        d_training_input, 
        static_cast<std::size_t>(m_tiles),
        static_cast<std::size_t>(n_tiles), 
        static_cast<std::size_t>(m_tile_size), 
        static_cast<std::size_t>(n_tile_size), 
        static_cast<std::size_t>(n_regressors),   
        sek_params, 
        sycl_device
    );

    std::cout << "[sycl_gp_functions.cpp] [predict] : Accessing assemble_tiles_with_zeros \n";

    auto prediction_tiles = assemble_tiles_with_zeros(static_cast<std::size_t>(m_tile_size), static_cast<std::size_t>(m_tiles), sycl_device);

    std::cout << "[sycl_gp_functions.cpp] [predict] : Accessing right_looking_cholesky_tiled \n";

    right_looking_cholesky_tiled(d_tiles, static_cast<std::size_t>(n_tile_size), static_cast<std::size_t>(n_tiles), sycl_device);

    // Triangular solve K_NxN * alpha = y

    std::cout << "[sycl_gp_functions.cpp] [predict] : Accessing forward_solve_tiled \n";

    forward_solve_tiled(d_tiles, alpha_tiles, static_cast<std::size_t>(n_tile_size), static_cast<std::size_t>(n_tiles), sycl_device);

    std::cout << "[sycl_gp_functions.cpp] [predict] : Accessing backward_solve_tiled \n";

    backward_solve_tiled(d_tiles, alpha_tiles, static_cast<std::size_t>(n_tile_size), static_cast<std::size_t>(n_tiles), sycl_device);

    std::cout << "[sycl_gp_functions.cpp] [predict] : Accessing matrix_vector_tiled \n";

    matrix_vector_tiled(
        cross_covariance_tiles, 
        alpha_tiles, 
        prediction_tiles, 
        static_cast<std::size_t>(m_tile_size), 
        static_cast<std::size_t>(n_tile_size),
        static_cast<std::size_t>(n_tiles), 
        static_cast<std::size_t>(m_tiles), 
        sycl_device
    );

    std::cout << "[sycl_gp_functions.cpp] [predict] : Accessing copy_tiled_vector_to_host_vector \n";

    std::vector<double> prediction = copy_tiled_vector_to_host_vector(
        prediction_tiles, 
        static_cast<std::size_t>(m_tile_size), 
        std::size_t(m_tiles), 
        sycl_device
    );

    std::cout << "[sycl_gp_functions.cpp] [predict] : Accessing gprat::sycl_backend::free_lower_tiled_matrix \n";

    gprat::sycl_backend::free_lower_tiled_matrix(d_tiles, static_cast<std::size_t>(n_tiles), sycl_device);

    sycl::queue queue = sycl_device.next_queue();

    std::cout << "[sycl_gp_functions.cpp] [predict] : Freeing the entire world \n";

    gprat::sycl_backend::free(alpha_tiles, queue);
    gprat::sycl_backend::free(cross_covariance_tiles, queue);
    gprat::sycl_backend::free(prediction_tiles, queue);
    
    std::cout << "[sycl_gp_functions.cpp] [predict] : Destroying SYCL device with dynamite \n";

    sycl_device.destroy();

    std::cout << "[sycl_gp_functions.cpp] [predict] : All done, returning \n";

    return prediction;
}

// predict_with_uncertainty ///////////////////////////////////////////////////////////////////////////////////////////

std::vector<std::vector<double>> 
predict_with_uncertainty(
    const std::vector<double> &h_training_input,
    const std::vector<double> &h_training_output,
    const std::vector<double> &h_test_input,
    const gprat_hyper::SEKParams &sek_params,
    int n_tiles,
    int n_tile_size,
    int m_tiles,
    int m_tile_size,
    int n_regressors,
    gprat::SYCL_DEVICE &sycl_device
)
{
    sycl_device.create();

    double *d_training_input = copy_to_device(h_training_input, sycl_device);
    double *d_training_output = copy_to_device(h_training_output, sycl_device);
    double *d_test_input = copy_to_device(h_test_input, sycl_device);

    // Assemble tiled covariance matrix on GPU.
    auto d_K_tiles = 
        assemble_tiled_covariance_matrix(
        d_training_input, 
        static_cast<std::size_t>(n_tiles), 
        static_cast<std::size_t>(n_tile_size),  
        static_cast<std::size_t>(n_regressors),   
        sek_params, 
        sycl_device
    );

    auto d_alpha_tiles = 
        assemble_alpha_tiles(d_training_output, static_cast<std::size_t>(n_tiles), static_cast<std::size_t>(n_tile_size), sycl_device);

    auto d_prior_K_tiles = 
        assemble_prior_K_tiles(d_test_input, static_cast<std::size_t>(m_tiles), static_cast<std::size_t>(m_tile_size), static_cast<std::size_t>(n_regressors), sek_params, sycl_device);

    auto d_cross_covariance_tiles = assemble_cross_covariance_tiles(
        d_test_input, 
        d_training_input, 
        static_cast<std::size_t>(m_tiles),
        static_cast<std::size_t>(n_tiles), 
        static_cast<std::size_t>(m_tile_size), 
        static_cast<std::size_t>(n_tile_size), 
        static_cast<std::size_t>(n_regressors),   
        sek_params, 
        sycl_device
    );

    auto d_t_cross_covariance_tiles = assemble_t_cross_covariance_tiles(
        d_cross_covariance_tiles, 
        static_cast<std::size_t>(n_tiles), 
        static_cast<std::size_t>(m_tiles), 
        static_cast<std::size_t>(n_tile_size), 
        static_cast<std::size_t>(m_tile_size), 
        sycl_device
    );

    // Assemble placeholder matrix for diag(K_MxN * (K^-1_NxN * K_NxM))
    auto d_prior_inter_tiles = assemble_tiles_with_zeros(static_cast<std::size_t>(m_tile_size), static_cast<std::size_t>(m_tiles), sycl_device);

    auto d_prediction_tiles = assemble_tiles_with_zeros(static_cast<std::size_t>(m_tile_size), static_cast<std::size_t>(m_tiles), sycl_device);

    // Assemble placeholder for uncertainty
    auto d_prediction_uncertainty_tiles = assemble_tiles_with_zeros(static_cast<std::size_t>(m_tile_size), static_cast<std::size_t>(m_tiles), sycl_device);

    right_looking_cholesky_tiled(d_K_tiles, static_cast<std::size_t>(n_tile_size), static_cast<std::size_t>(n_tiles), sycl_device);

    // Triangular solve K_NxN * alpha = y
    forward_solve_tiled(d_K_tiles, d_alpha_tiles, static_cast<std::size_t>(n_tile_size), static_cast<std::size_t>(n_tiles), sycl_device);
    backward_solve_tiled(d_K_tiles, d_alpha_tiles, static_cast<std::size_t>(n_tile_size), static_cast<std::size_t>(n_tiles), sycl_device);

    // Triangular solve A_M,N * K_NxN = K_MxN -> A_MxN = K_MxN * K^-1_NxN
    forward_solve_tiled_matrix(
        d_K_tiles, 
        d_t_cross_covariance_tiles, 
        static_cast<std::size_t>(n_tile_size), 
        static_cast<std::size_t>(m_tile_size), 
        static_cast<std::size_t>(n_tiles),
        static_cast<std::size_t>(m_tiles), 
        sycl_device
    );

    // Compute predictions
    matrix_vector_tiled(
        d_cross_covariance_tiles,
        d_alpha_tiles, 
        d_prediction_tiles, 
        static_cast<std::size_t>(m_tile_size), 
        static_cast<std::size_t>(n_tile_size), 
        static_cast<std::size_t>(n_tiles), 
        static_cast<std::size_t>(m_tiles), 
        sycl_device
    );

    // posterior covariance matrix - (K_MxN * K^-1_NxN) * K_NxM
    symmetric_matrix_matrix_diagonal_tiled(
        d_t_cross_covariance_tiles, d_prior_inter_tiles, static_cast<std::size_t>(n_tile_size), static_cast<std::size_t>(m_tile_size), static_cast<std::size_t>(n_tiles), static_cast<std::size_t>(m_tiles), sycl_device);

    // Compute predicition uncertainty
    vector_difference_tiled(
        d_prior_K_tiles, d_prior_inter_tiles, d_prediction_uncertainty_tiles, static_cast<std::size_t>(m_tile_size), static_cast<std::size_t>(m_tiles), sycl_device);

    // Get predictions and uncertainty to return them
    std::vector<double> prediction = 
        copy_tiled_vector_to_host_vector(d_prediction_tiles, static_cast<std::size_t>(m_tile_size), static_cast<std::size_t>(m_tiles), sycl_device);

    std::vector<double> pred_var_full =
        copy_tiled_vector_to_host_vector(d_prediction_uncertainty_tiles, static_cast<std::size_t>(m_tile_size), static_cast<std::size_t>(m_tiles), sycl_device);

    sycl::queue queue = sycl_device.next_queue();
    sycl::free(d_training_input, queue);
    sycl::free(d_training_output, queue);
    sycl::free(d_test_input, queue);

    gprat::sycl_backend::free_lower_tiled_matrix(d_K_tiles, static_cast<std::size_t>(n_tiles), sycl_device);

    gprat::sycl_backend::free(d_alpha_tiles, queue);
    gprat::sycl_backend::free(d_prior_K_tiles, queue);
    gprat::sycl_backend::free(d_cross_covariance_tiles, queue);
    gprat::sycl_backend::free(d_t_cross_covariance_tiles, queue);
    gprat::sycl_backend::free(d_prior_inter_tiles, queue);
    gprat::sycl_backend::free(d_prediction_tiles, queue);
    gprat::sycl_backend::free(d_prediction_uncertainty_tiles, queue);

    sycl_device.destroy();

    return std::vector<std::vector<double>>{ prediction, pred_var_full };
}

// predict_with_full_cov //////////////////////////////////////////////////////////////////////////////////////////////

std::vector<std::vector<double>> 
predict_with_full_cov(
    const std::vector<double> &h_training_input,
    const std::vector<double> &h_training_output,
    const std::vector<double> &h_test_input,
    const gprat_hyper::SEKParams &sek_params,
    int n_tiles,
    int n_tile_size,
    int m_tiles,
    int m_tile_size,
    int n_regressors,
    gprat::SYCL_DEVICE &sycl_device
)
{
    sycl_device.create();

    double *d_training_input = copy_to_device(h_training_input, sycl_device);
    double *d_training_output = copy_to_device(h_training_output, sycl_device);
    double *d_test_input = copy_to_device(h_test_input, sycl_device);

    // Assemble tiled covariance matrix on GPU.
    auto d_K_tiles = assemble_tiled_covariance_matrix(
        d_training_input, 
        static_cast<std::size_t>(n_tiles), 
        static_cast<std::size_t>(n_tile_size),  
        static_cast<std::size_t>(n_regressors),   
        sek_params, 
        sycl_device
    );

    auto d_alpha_tiles = assemble_alpha_tiles(d_training_output, static_cast<std::size_t>(n_tiles), static_cast<std::size_t>(n_tile_size), sycl_device);

    auto d_prior_K_tiles =
        assemble_prior_K_tiles_full(d_test_input, static_cast<std::size_t>(m_tiles), static_cast<std::size_t>(m_tile_size), static_cast<std::size_t>(n_regressors), sek_params, sycl_device);

    auto d_cross_covariance_tiles = assemble_cross_covariance_tiles(
        d_test_input, 
        d_training_input, 
        static_cast<std::size_t>(m_tiles), 
        static_cast<std::size_t>(n_tiles), 
        static_cast<std::size_t>(m_tile_size), 
        static_cast<std::size_t>(n_tile_size), 
        static_cast<std::size_t>(n_regressors),   
        sek_params, 
        sycl_device
    );

    auto d_t_cross_covariance_tiles = assemble_t_cross_covariance_tiles(
        d_cross_covariance_tiles, 
        static_cast<std::size_t>(n_tiles), 
        static_cast<std::size_t>(m_tiles), 
        static_cast<std::size_t>(n_tile_size), 
        static_cast<std::size_t>(m_tile_size), 
        sycl_device
    );

    auto d_prediction_tiles = assemble_tiles_with_zeros(static_cast<std::size_t>(m_tile_size), static_cast<std::size_t>(m_tiles), sycl_device);

    // Assemble placeholder for uncertainty
    auto d_prediction_uncertainty_tiles = assemble_tiles_with_zeros(static_cast<std::size_t>(m_tile_size), static_cast<std::size_t>(m_tiles), sycl_device);


    right_looking_cholesky_tiled(d_K_tiles, static_cast<std::size_t>(n_tile_size), static_cast<std::size_t>(n_tiles), sycl_device);

    // Triangular solve K_NxN * alpha = y
    forward_solve_tiled(d_K_tiles, d_alpha_tiles, static_cast<std::size_t>(n_tile_size), static_cast<std::size_t>(n_tiles), sycl_device);
    backward_solve_tiled(d_K_tiles, d_alpha_tiles, static_cast<std::size_t>(n_tile_size), static_cast<std::size_t>(n_tiles), sycl_device);

    // Triangular solve A_M,N * K_NxN = K_MxN -> A_MxN = K_MxN * K^-1_NxN
    forward_solve_tiled_matrix(
        d_K_tiles, 
        d_t_cross_covariance_tiles, 
        static_cast<std::size_t>(n_tile_size), 
        static_cast<std::size_t>(m_tile_size), 
        static_cast<std::size_t>(n_tiles),
        static_cast<std::size_t>(m_tiles), 
        sycl_device
    );

    // Compute predictions
    matrix_vector_tiled(
        d_cross_covariance_tiles, 
        d_alpha_tiles, 
        d_prediction_tiles, 
        static_cast<std::size_t>(m_tile_size), 
        static_cast<std::size_t>(n_tile_size), 
        static_cast<std::size_t>(n_tiles),
        static_cast<std::size_t>(m_tiles), 
        sycl_device
    );

    // posterior covariance matrix - (K_MxN * K^-1_NxN) * K_NxM
    symmetric_matrix_matrix_tiled(
        d_t_cross_covariance_tiles, d_prior_K_tiles, static_cast<std::size_t>(n_tile_size), static_cast<std::size_t>(m_tile_size), static_cast<std::size_t>(n_tiles), static_cast<std::size_t>(m_tiles), sycl_device);

    // Compute predicition uncertainty
    matrix_diagonal_tiled(d_prior_K_tiles, d_prediction_uncertainty_tiles, static_cast<std::size_t>(m_tile_size), static_cast<std::size_t>(m_tiles), sycl_device);

    // Get predictions and uncertainty to return them
    std::vector<double> prediction = 
        copy_tiled_vector_to_host_vector(d_prediction_tiles, static_cast<std::size_t>(m_tile_size), static_cast<std::size_t>(m_tiles), sycl_device);
    std::vector<double> pred_var_full =
        copy_tiled_vector_to_host_vector(d_prediction_uncertainty_tiles, static_cast<std::size_t>(m_tile_size), static_cast<std::size_t>(m_tiles), sycl_device);

    sycl::queue queue = sycl_device.next_queue();

    sycl::free(d_training_input, queue);
    sycl::free(d_training_output, queue);
    sycl::free(d_test_input, queue);

    gprat::sycl_backend::free_lower_tiled_matrix(d_K_tiles, static_cast<std::size_t>(n_tiles), sycl_device);

    gprat::sycl_backend::free(d_alpha_tiles, queue);

    gprat::sycl_backend::free_lower_tiled_matrix(d_prior_K_tiles, static_cast<std::size_t>(m_tiles), sycl_device);
    gprat::sycl_backend::free(d_cross_covariance_tiles, queue);
    gprat::sycl_backend::free(d_t_cross_covariance_tiles, queue);
    gprat::sycl_backend::free(d_prediction_tiles, queue);
    gprat::sycl_backend::free(d_prediction_uncertainty_tiles, queue);
    

    sycl_device.destroy();

    return std::vector<std::vector<double>>{ prediction, pred_var_full };
}

// compute_loss ///////////////////////////////////////////////////////////////////////////////////////////////////////

double 
compute_loss(
    const std::vector<double> &h_training_input,    
    const std::vector<double> &h_training_output,           
    const gprat_hyper::SEKParams &sek_params,            
    int n_tiles,
    int n_tile_size,
    int n_regressors,
    gprat::SYCL_DEVICE &sycl_device
)
{
    sycl_device.create();

    double *d_training_input = copy_to_device(h_training_input, sycl_device);
    double *d_training_output = copy_to_device(h_training_output, sycl_device);

    // Assemble tiled covariance matrix on GPU.
    auto d_K_tiles = assemble_tiled_covariance_matrix(
        d_training_input, 
        static_cast<std::size_t>(n_tiles), 
        static_cast<std::size_t>(n_tile_size),  
        static_cast<std::size_t>(n_regressors),   
        sek_params, 
        sycl_device
    );

    auto d_alpha_tiles = assemble_alpha_tiles(d_training_output, static_cast<std::size_t>(n_tiles), static_cast<std::size_t>(n_tile_size), sycl_device);

    auto d_y_tiles = assemble_y_tiles(d_training_output, static_cast<std::size_t>(n_tiles), static_cast<std::size_t>(n_tile_size), sycl_device);

    right_looking_cholesky_tiled(d_K_tiles, static_cast<std::size_t>(n_tile_size), static_cast<std::size_t>(n_tiles), sycl_device);

    // Triangular solve K_NxN * alpha = y
    forward_solve_tiled(d_K_tiles, d_alpha_tiles, static_cast<std::size_t>(n_tile_size), static_cast<std::size_t>(n_tiles), sycl_device);
    backward_solve_tiled(d_K_tiles, d_alpha_tiles, static_cast<std::size_t>(n_tile_size), static_cast<std::size_t>(n_tiles), sycl_device);

    // Compute loss
    hpx::shared_future<double> loss_value =
        compute_loss_tiled(d_K_tiles, d_alpha_tiles, d_y_tiles, static_cast<std::size_t>(n_tile_size), static_cast<std::size_t>(n_tiles), sycl_device);

    sycl::queue queue = sycl_device.next_queue();

    sycl::free(d_training_input, queue);
    sycl::free(d_training_output, queue);

    loss_value.get();

    gprat::sycl_backend::free_lower_tiled_matrix(d_K_tiles, static_cast<std::size_t>(n_tiles), sycl_device);

    gprat::sycl_backend::free(d_alpha_tiles, queue);
    gprat::sycl_backend::free(d_y_tiles, queue);
    
    sycl_device.destroy();

    return loss_value.get();
}

// optimize ///////////////////////////////////////////////////////////////////////////////////////////////////////////

std::vector<double>
optimize(
    // const std::vector<double> &training_input,
    // const std::vector<double> &training_output,
    // int n_tiles,
    // int n_tile_size,
    // int n_regressors,
    // const gprat_hyper::AdamParams &adam_params,
    // const gprat_hyper::SEKParams &sek_params,
    // std::vector<bool> trainable_params,
    // gprat::SYCL_DEVICE &sycl_device
)
{
    throw std::logic_error("Function not implemented for GPU");
    // return std::vector<double>>();
}

// optimize_step //////////////////////////////////////////////////////////////////////////////////////////////////////

double 
optimize_step(
    // const std::vector<double> &training_input,
    // const std::vector<double> &training_output,
    // int n_tiles,
    // int n_tile_size,
    // int n_regressors,
    // gprat_hyper::AdamParams &adam_params,
    // gprat_hyper::SEKParams &sek_params,
    // std::vector<bool> trainable_params,
    // int iter,
    // gprat::SYCL_DEVICE &sycl_device
)
{
    throw std::logic_error("Function not implemented for GPU");
    // return 0.0;
}

// cholesky ///////////////////////////////////////////////////////////////////////////////////////////////////////////

std::vector<std::vector<double>>
cholesky(
    const std::vector<double> &h_training_input,
    const gprat_hyper::SEKParams &sek_params,
    int n_tiles,
    int n_tile_size,
    int n_regressors,
    gprat::SYCL_DEVICE &sycl_device
)
{
    sycl_device.create();

    double *d_training_input = copy_to_device(h_training_input, sycl_device);

    // Assemble tiled covariance matrix on GPU.
    std::vector<hpx::shared_future<double *>> d_tiles = assemble_tiled_covariance_matrix(
        d_training_input, 
        static_cast<std::size_t>(n_tiles), 
        static_cast<std::size_t>(n_tile_size),  
        static_cast<std::size_t>(n_regressors),   
        sek_params, 
        sycl_device
    );

    // Compute Tiled Cholesky decomposition on device
    right_looking_cholesky_tiled(d_tiles, static_cast<std::size_t>(n_tile_size), static_cast<std::size_t>(n_tiles), sycl_device);

    // Copy tiled matrix to host
    std::vector<std::vector<double>> h_tiles = move_lower_tiled_matrix_to_host(
        d_tiles, 
        static_cast<std::size_t>(n_tile_size), 
        static_cast<std::size_t>(n_tiles), 
        sycl_device
    );

    sycl::queue queue = sycl_device.next_queue();
    sycl::free(d_training_input, queue);
    
    sycl_device.destroy();

    return h_tiles;
}

}  // end of namespace sycl_backend
