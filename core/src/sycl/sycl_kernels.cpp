// Includes ///////////////////////////////////////////////////////////////////////////////////////////////////////////

#include "sycl/sycl_kernels.hpp"
#include "sycl/sycl_utils.hpp"

// Transpose kernel ///////////////////////////////////////////////////////////////////////////////////////////////////

TransposeKernel::TransposeKernel(
    double *transposed, 
    double *original, 
    std::size_t width, 
    std::size_t height, 
    sycl::handler &cgh
) :
transposed(transposed),
original(original),
width(width),
height(height),
local(sycl::local_accessor<gprat::sycl_backend::real_t, 2>(sycl::range<2>(WORK_GROUP_SIZE, WORK_GROUP_SIZE + 1), cgh))
{}

void TransposeKernel::operator()(const sycl::nd_item<2> &nd_item) const
{
    // Local indices
    const std::size_t local_x = nd_item.get_local_id(1);
    const std::size_t local_y = nd_item.get_local_id(0);

    // Group indices
    const std::size_t group_x = nd_item.get_group(1);
    const std::size_t group_y = nd_item.get_group(0);

    // Global coordinates (like CUDA's blockIdx * blockDim + threadIdx)
    std::size_t xIndex = group_x * WORK_GROUP_SIZE + local_x;
    std::size_t yIndex = group_y * WORK_GROUP_SIZE + local_y;

    // Load tile into local memory
    if (xIndex < width && yIndex < height)
    {
        local[local_y][local_x] = original[yIndex * width + xIndex];
    }
    else
    {
        local[local_y][local_x] = 0.0; // optional padding
    }

    nd_item.barrier(sycl::access::fence_space::local_space);

    // Compute swapped indices for transpose write-back
    std::size_t x_index_out = group_y * WORK_GROUP_SIZE + local_x;
    std::size_t y_index_out = group_x * WORK_GROUP_SIZE + local_y;

    if (x_index_out < height && y_index_out < width)
    {
        transposed[y_index_out * height + x_index_out] = local[local_x][local_y];
    }
}

// GenTileCovarianceKernel ////////////////////////////////////////////////////////////////////////////////////////////

GenTileCovarianceKernel::GenTileCovarianceKernel(
    double *d_tile,
    const double *d_input,
    const std::size_t n_tile_size,
    const std::size_t n_regressors,
    const std::size_t tile_row,
    const std::size_t tile_column,
    const gprat_hyper::SEKParams sek_params
) :
d_tile(d_tile),
d_input(d_input),
n_tile_size(n_tile_size),
n_regressors(n_regressors),
tile_row(tile_row),
tile_column(tile_column),
lengthscale_(sek_params.lengthscale),
vertical_lengthscale_(sek_params.vertical_lengthscale),
noise_variance_(sek_params.noise_variance)
{}

// Done, check if compiles
// Launch over range [0, n_tile_size] x [0, n_tile_size]
void GenTileCovarianceKernel::operator()(const sycl::item<2> &item) const
{
    const std::size_t i = item.get_id(0);
    const std::size_t j = item.get_id(1);

    const std::size_t i_global = n_tile_size * tile_row + i;
    const std::size_t j_global = n_tile_size * tile_column + j;

    double distance = 0.0;
    double z_ik_minus_z_jk = 0.0;

    for (std::size_t k = 0; k < n_regressors; ++k)
    {
        z_ik_minus_z_jk = d_input[i_global + k] - d_input[j_global + k];
        distance += z_ik_minus_z_jk * z_ik_minus_z_jk;
    }

    double covariance =
        vertical_lengthscale_ *
        sycl::exp(-0.5 * distance / (lengthscale_ * lengthscale_));

    if (i_global == j_global) { covariance += noise_variance_; }

    d_tile[i * n_tile_size + j] = covariance;
}

// GenTileFullPriorCovarianceKernel ///////////////////////////////////////////////////////////////////////////////////

GenTileFullPriorCovarianceKernel::GenTileFullPriorCovarianceKernel(
    double *d_tile,
    const double *d_input,
    const std::size_t n_tile_size,
    const std::size_t n_regressors,
    const std::size_t tile_row,
    const std::size_t tile_column,
    const gprat_hyper::SEKParams sek_params
) :
d_tile(d_tile),
d_input(d_input),
n_tile_size(n_tile_size),
n_regressors(n_regressors),
tile_row(tile_row),
tile_column(tile_column),
lengthscale_(sek_params.lengthscale),
vertical_lengthscale_(sek_params.vertical_lengthscale)
{}

// Done, check if compiles
// Launch over range [0, n_tile_size] x [0, n_tile_size]
void GenTileFullPriorCovarianceKernel::operator()(const sycl::item<2> &item) const
{
    const std::size_t i = item.get_id(0);
    const std::size_t j = item.get_id(1);

    const std::size_t i_global = n_tile_size * tile_row + i;
    const std::size_t j_global = n_tile_size * tile_column + j;

    double distance = 0.0;
    double z_ik_minus_z_jk = 0.0;

    for (std::size_t k = 0; k < n_regressors; ++k)
    {
        z_ik_minus_z_jk = d_input[i_global + k] - d_input[j_global + k];
        distance += z_ik_minus_z_jk * z_ik_minus_z_jk;
    }

    const double covariance =
        vertical_lengthscale_ * 
        sycl::exp(-0.5 * distance / (lengthscale_ * lengthscale_));

    d_tile[i * n_tile_size + j] = covariance;
}

// GenTilePriorCovarianceKernel ///////////////////////////////////////////////////////////////////////////////////////

GenTilePriorCovarianceKernel::GenTilePriorCovarianceKernel(
    double *d_tile,
    const double *d_input,
    const std::size_t n_tile_size,
    const std::size_t n_regressors,
    const std::size_t tile_row,
    const std::size_t tile_column,
    const gprat_hyper::SEKParams sek_params
) :
d_tile(d_tile),
d_input(d_input),
n_tile_size(n_tile_size),
n_regressors(n_regressors),
tile_row(tile_row),
tile_column(tile_column),
lengthscale_(sek_params.lengthscale),
vertical_lengthscale_(sek_params.vertical_lengthscale)
{}

// Done, check if compiles
// Launch over linear kernel with range [0, n_tile_size]
void GenTilePriorCovarianceKernel::operator()(const sycl::id<1> &id) const
{
    std::size_t i_global = n_tile_size * tile_row + id;
    std::size_t j_global = n_tile_size * tile_column + id;

    double distance = 0.0;
    double z_ik_minus_z_jk = 0.0;

    for (std::size_t k = 0; k < n_regressors; ++k)
    {
        z_ik_minus_z_jk = d_input[i_global + k] - d_input[j_global + k];
        distance += z_ik_minus_z_jk * z_ik_minus_z_jk;
    }

    double covariance =
        vertical_lengthscale_ * exp(-0.5 * distance / (lengthscale_ * lengthscale_));

    d_tile[id] = covariance;
}

// GenTileCrossCovarianceKernel ///////////////////////////////////////////////////////////////////////////////////////

GenTileCrossCovarianceKernel::GenTileCrossCovarianceKernel(
    double *d_tile,
    const double *d_row_input,
    const double *d_col_input,
    const std::size_t n_row_tile_size,
    const std::size_t n_column_tile_size,
    const std::size_t tile_row,
    const std::size_t tile_column,
    const std::size_t n_regressors,
    const gprat_hyper::SEKParams sek_params
) :
d_tile(d_tile),
d_row_input(d_row_input),
d_col_input(d_col_input),
n_row_tile_size(n_row_tile_size),
n_column_tile_size(n_column_tile_size),
tile_row(tile_row),
tile_column(tile_column),
n_regressors(n_regressors),
lengthscale_(sek_params.lengthscale),
vertical_lengthscale_(sek_params.vertical_lengthscale)
{}

// Done, check if compiles
// Launch over range [0, n_row_tile_size] x [0, n_column_tile_size]
void GenTileCrossCovarianceKernel::operator()(const sycl::item<2> &item) const
{
    const std::size_t i = item.get_id(0);
    const std::size_t j = item.get_id(1);

    const std::size_t i_global = n_row_tile_size * tile_row + i;
    const std::size_t j_global = n_column_tile_size * tile_column + j;

    double distance = 0.0;
    double z_ik_minus_z_jk = 0.0;

    for (std::size_t k = 0; k < n_regressors; ++k)
    {
        z_ik_minus_z_jk = d_row_input[i_global + k] - d_col_input[j_global + k];
        distance += z_ik_minus_z_jk * z_ik_minus_z_jk;
    }

    const double covariance = vertical_lengthscale_ *
        sycl::exp(-0.5 * distance /(lengthscale_ * lengthscale_));

    d_tile[i * n_column_tile_size + j] = covariance;
}

// GenTileOutputKernel ////////////////////////////////////////////////////////////////////////////////////////////////

GenTileOutputKernel::GenTileOutputKernel(
    double *tile, 
    const double *output, 
    std::size_t row, 
    std::size_t n_tile_size
) : 
tile(tile),
output(output),
row(row),
n_tile_size(n_tile_size)
{}

// Done, check if compiles
// Launch over linear kernel with range [0, n_tile_size]
void GenTileOutputKernel::operator()(const sycl::id<1> &id) const
{
    std::size_t i_global = n_tile_size * row + id;
    tile[id] = output[i_global];
}

