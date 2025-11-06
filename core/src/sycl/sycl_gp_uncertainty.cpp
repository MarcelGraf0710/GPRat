#include "sycl/sycl_gp_uncertainty.hpp"
#include "sycl/sycl_utils.hpp"
#include "target.hpp"

namespace gprat::sycl_backend
{

hpx::shared_future<double *> diag_posterior(
    const hpx::shared_future<double *> A, 
    const hpx::shared_future<double *> B, 
    std::size_t M, 
    gprat::SYCL_DEVICE &sycl_device
)
{
    sycl::queue queue = sycl_device.next_queue();

    double *tile = sycl::malloc_device<gprat::sycl_backend::real_t>(M, queue);

    // tile = 1.0*A + (-1.0)*B
    oneapi::math::blas::column_major::omatadd(
        queue,
        oneapi::math::transpose::nontrans,
        oneapi::math::transpose::nontrans,
        1,
        M,
        1.0,
        A.get(),
        1,
        -1.0,
        B.get(),
        1,
        tile,
        1
    );

    queue.wait();

    return hpx::make_ready_future(tile);
}

hpx::shared_future<double *> diag_tile(
    const hpx::shared_future<double *> A, 
    std::size_t M, 
    gprat::SYCL_DEVICE &sycl_device
)
{
    sycl::queue queue = sycl_device.next_queue();

    double *diag_tile = sycl::malloc_device<gprat::sycl_backend::real_t>(M, queue);

    oneapi::math::blas::column_major::omatcopy(
        queue,
        oneapi::math::transpose::nontrans,
        1,                                  
        M,                                  
        1.0,
        A.get(),
        M + 1,
        diag_tile,
        1 
    );

    queue.wait();

    return hpx::make_ready_future(diag_tile);
}

}  // end of namespace sycl_backend
