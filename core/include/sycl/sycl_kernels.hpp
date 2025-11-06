#ifndef GPRAT_SYCL_KERNELS_H
#define GPRAT_SYCL_KERNELS_H

#include "sycl_utils.hpp"
#include "gp_kernels.hpp"

#include <cstddef>

/**
 * @brief Kernel to transpose a matrix.
 */
class TransposeKernel
{
    private:

    double *transposed;
    double *original;
    std::size_t width;
    std::size_t height;
    sycl::local_accessor<gprat::sycl_backend::real_t, 2> local;

    public:

    /**
     * @brief Construct a TransposeKernel object
     * 
     * @param transposed Pointer to the transposed output matrix.
     * @param original Pointer to the original input matrix.
     * @param width Width of the original matrix.
     * @param height Height of the original matrix.
     * @param cgh SYCL command group handler for local memory
     */
    explicit TransposeKernel(
        double *transposed, 
        double *original, 
        std::size_t width, 
        std::size_t height, 
        sycl::handler &cgh
    );

    void operator()(const sycl::nd_item<2> &nd_item) const SYCL_EXTERNAL;
};

/**
 * @brief Kernel function to compute covariance
 */
class GenTileCovarianceKernel 
{
    private:

    double *d_tile;
    const double *d_input;
    std::size_t n_tile_size;
    std::size_t n_regressors;
    std::size_t tile_row;
    std::size_t tile_column;
    double lengthscale_;
    double vertical_lengthscale_;
    double noise_variance_;

    public:

    /**
     * @brief Generate a tile of the covariance matrix
     */
    explicit GenTileCovarianceKernel(
        double *d_tile,
        const double *d_input,
        const std::size_t n_tile_size,
        const std::size_t n_regressors,
        const std::size_t tile_row,
        const std::size_t tile_column,
        const gprat_hyper::SEKParams sek_params
    );

    void operator()(const sycl::item<2> &item) const SYCL_EXTERNAL;
};

/**
 * @brief 
 * 
 */
class GenTileFullPriorCovarianceKernel
{
    private:

    double *d_tile;
    const double *d_input;
    std::size_t n_tile_size;
    std::size_t n_regressors;
    std::size_t tile_row;
    std::size_t tile_column;
    double lengthscale_;
    double vertical_lengthscale_;

    public:

    explicit GenTileFullPriorCovarianceKernel(
        double *d_tile,
        const double *d_input,
        const std::size_t n_tile_size,
        const std::size_t n_regressors,
        const std::size_t tile_row,
        const std::size_t tile_column,
        const gprat_hyper::SEKParams sek_params
    );

    void operator()(const sycl::item<2> &item) const SYCL_EXTERNAL;
};

/**
 * @brief 
 * 
 */
class GenTilePriorCovarianceKernel
{
    private:

    double *d_tile;
    const double *d_input;
    std::size_t n_tile_size;
    std::size_t n_regressors;
    std::size_t tile_row;
    std::size_t tile_column;
    double lengthscale_;
    double vertical_lengthscale_;

    public:

    explicit GenTilePriorCovarianceKernel(
        double *d_tile,
        const double *d_input,
        const std::size_t n_tile_size,
        const std::size_t n_regressors,
        const std::size_t tile_row,
        const std::size_t tile_column,
        const gprat_hyper::SEKParams sek_params
    );

    void operator()(const sycl::id<1> &id) const SYCL_EXTERNAL;
};

/**
 * @brief 
 * 
 */
class GenTileCrossCovarianceKernel
{
    private:

    double *d_tile;
    const double *d_row_input;
    const double *d_col_input;
    std::size_t n_row_tile_size;
    std::size_t n_column_tile_size;
    std::size_t tile_row;
    std::size_t tile_column;
    std::size_t n_regressors;
    double lengthscale_;
    double vertical_lengthscale_;

    public:

    explicit GenTileCrossCovarianceKernel(
        double *d_tile,
        const double *d_row_input,
        const double *d_col_input,
        const std::size_t n_row_tile_size,
        const std::size_t n_column_tile_size,
        const std::size_t tile_row,
        const std::size_t tile_column,
        const std::size_t n_regressors,
        const gprat_hyper::SEKParams sek_params
    );

    void operator()(const sycl::item<2> &item) const SYCL_EXTERNAL;
};

/**
 * @brief 
 * 
 */
class GenTileOutputKernel
{
    private:

    double *tile;
    const double *output; 
    std::size_t row; 
    std::size_t n_tile_size;

    public:

    explicit GenTileOutputKernel(
        double *tile, 
        const double *output, 
        std::size_t row, 
        std::size_t n_tile_size
    );

    void operator()(const sycl::id<1> &id) const SYCL_EXTERNAL;
};

#endif // ! GPRAT_SYCL_KERNELS_H
