#ifndef SYCL_GP_UNCERTAINTY_H
#define SYCL_GP_UNCERTAINTY_H

#include <hpx/future.hpp>
#include "target.hpp"

#include <oneapi/math.hpp>

namespace gprat::sycl_backend
{

/**
 * @brief Retrieve diagonal elements of posterior covariance matrix.
 *
 * @param A Diagonal elements matrix A
 * @param B Diagonal elements matrix B
 * @param M Number of rows in the matrix
 *
 * @return Diagonal elements of posterior covariance matrix
 */
hpx::shared_future<double *> diag_posterior(
    const hpx::shared_future<double *> A, const hpx::shared_future<double *> B, std::size_t M, gprat::SYCL_DEVICE &sycl_device);

/**
 * @brief Retrieve diagonal elements of posterior covariance matrix.
 *
 * @param A Posterior covariance matrix
 * @param M Number of rows in the matrix
 *
 * @return Diagonal elements of posterior covariance matrix
 */
hpx::shared_future<double *> diag_tile(const hpx::shared_future<double *> A, std::size_t M, gprat::SYCL_DEVICE &sycl_device);

}  // end of namespace gpu

#endif  // end of SYCL_GP_UNCERTAINTY_H
