#include "sycl/adapter_onemath.hpp"

// BLAS LEVEL 3 OPERATIONS ////////////////////////////////////////////////////////////////////////////////////////////

hpx::shared_future<double *>
potrf(sycl::queue &queue, hpx::shared_future<double *> f_A, const std::size_t N)
{
    std::int64_t scratchpad_size = oneapi::math::lapack::potrf_scratchpad_size<double>(
        queue, 
        oneapi::math::uplo::upper, 
        static_cast<std::int64_t>(N), 
        static_cast<std::int64_t>(N)
    );

    gprat::sycl_backend::real_t *scratchpad = sycl::malloc_shared<gprat::sycl_backend::real_t>(static_cast<std::size_t>(scratchpad_size), queue);

    gprat::sycl_backend::real_t *d_A = f_A.get();

    // row-major POTRF
    // A = potrf(A)
    // for LOWER part of symmetric positive semi-definite matrix A

    // column-major cuBLAS POTRF for row-major stored A
    // for UPPER part of symmetric positive semi-definite matrix A

    oneapi::math::lapack::potrf(queue, oneapi::math::uplo::upper, static_cast<std::int64_t>(N), d_A, static_cast<std::int64_t>(N), scratchpad, scratchpad_size);
        
    queue.wait();

    sycl::free(scratchpad, queue);

    return hpx::make_ready_future(d_A);
}


hpx::shared_future<double *>
trsm(sycl::queue &queue,
     hpx::shared_future<double *> f_A,
     hpx::shared_future<double *> f_B,
     const std::size_t M,
     const std::size_t N,
     const oneapi::math::transpose is_transposed,
     const oneapi::math::side is_right)
{
    // TRSM constants
    const double alpha = 1.0;
    double *d_A = f_A.get();
    double *d_B = f_B.get();

    // row-major TRSM solves for X
    //
    // for side_A == Blas_right:
    //   op(A) * X = alpha * B
    //     A^T * X = B
    //
    // for side_A == Blas_left:
    //   X * op(A) = alpha * B
    //     X * A^T = B
    //
    // for op: transpose_A

    // column-major cuBLAS TRSM for row-major stored A & B
    // for X on opposite side (opposite of side_A)

    oneapi::math::blas::column_major::trsm(
        queue,
        invert_side_operator(is_right),
        oneapi::math::uplo::upper,
        is_transposed,
        oneapi::math::diag::nonunit,
        static_cast<std::int64_t>(M),
        static_cast<std::int64_t>(N),
        alpha,
        d_A,
        static_cast<std::int64_t>(M),
        d_B,
        static_cast<std::int64_t>(N));

    queue.wait();
    
    return hpx::make_ready_future(d_B);
}

hpx::shared_future<double *>
syrk(sycl::queue &queue,
     hpx::shared_future<double *> f_A,
     hpx::shared_future<double *> f_C,
     const std::size_t N)
{
    // SYRK constants
    const double alpha = -1.0;
    const double beta = 1.0;
    double *d_A = f_A.get();
    double *d_C = f_C.get();

    // row-major SYRK
    // C = alpha * op(A) * op(A)^T + beta * C
    //     C = - A * A^T + C
    // for LOWER part of symmetric matrix C
    // for op: NO transpose:

    // column-major cuBLAS SYRK for row-major stored A & C
    // C = - op(A) * op(A)^T + fm(C)
    //   = - A^T * A - C
    // for UPPER part of symmetric matrix C
    // for op: TRANSPOSE

    oneapi::math::blas::column_major::syrk(
        queue,
        oneapi::math::uplo::upper,
        oneapi::math::transpose::trans,
        static_cast<std::int64_t>(N),
        static_cast<std::int64_t>(N),
        alpha,
        d_A,
        static_cast<std::int64_t>(N),
        beta,
        d_C,
        static_cast<std::int64_t>(N));

    queue.wait();

    return hpx::make_ready_future(d_C);
}

hpx::shared_future<double *>
gemm(sycl::queue &queue,
     hpx::shared_future<double *> f_A,
     hpx::shared_future<double *> f_B,
     hpx::shared_future<double *> f_C,
     const std::size_t M,
     const std::size_t N,
     const std::size_t K,
     const oneapi::math::transpose is_A_transposed,
     const oneapi::math::transpose is_B_transposed)
{
    const double alpha = -1.0;
    const double beta = 1.0;
    double *d_A = f_A.get();
    double *d_B = f_B.get();
    double *d_C = f_C.get();

    // row-major GEMM
    // C = alpha * op(A) * op(B) + beta * C
    //   = op(A) * op(B) - C
    // for op(A): transpose_A
    // for op(B): transpose_B

    // column-major cuBLAS GEMM for row-major stored A, B, C
    // C = alpha * op(B) * op(A) + beta * C
    //   = op(B) * op(A) - C
    // for inverted ordering of matrices A, B

    oneapi::math::blas::column_major::gemm(
        queue,
        is_B_transposed,
        is_A_transposed,
        static_cast<std::int64_t>(N),
        static_cast<std::int64_t>(M),
        static_cast<std::int64_t>(K),
        alpha,
        d_B,
        static_cast<std::int64_t>(N),
        d_A,
        static_cast<std::int64_t>(K),
        beta,
        d_C,
        static_cast<std::int64_t>(N)); 

    queue.wait();

    return hpx::make_ready_future(d_C);
}

// BLAS LEVEL 2 OPERATIONS ////////////////////////////////////////////////////////////////////////////////////////////

hpx::shared_future<double *>
trsv(sycl::queue &queue,
     hpx::shared_future<double *> f_A,
     hpx::shared_future<double *> f_b,
     const std::size_t N,
     const oneapi::math::transpose is_A_transposed)
{
    double *d_A = f_A.get();
    double *d_b = f_b.get();

    // row-major TRSV solves for x
    // op(A) * x = b
    // for op: transpose_A
    // for LOWER part of lower triangular matrix A

    // column-major cuBLAS TRSV for row-major stored A
    // for op: opposite of transpose_A
    // for UPPER part of lower triangular matrix A

    oneapi::math::blas::column_major::trsv(
        queue,
        oneapi::math::uplo::upper,
        invert_transpose_operator(is_A_transposed),
        oneapi::math::diag::nonunit,
        static_cast<std::int64_t>(N),
        d_A,
        static_cast<std::int64_t>(N),
        d_b,
        1);

    queue.wait();

    // return solution vector x
    return hpx::make_ready_future(d_b);
}

hpx::shared_future<double *>
gemv(sycl::queue &queue,
     hpx::shared_future<double *> f_A,
     hpx::shared_future<double *> f_x,
     hpx::shared_future<double *> f_y,
     const std::size_t M,
     const std::size_t N,
     const double alpha,
     const oneapi::math::transpose is_A_transposed)
{
    auto d_A = f_A.get();
    auto d_x = f_x.get();
    auto d_y = f_y.get();

    const double alpha_value = alpha;
    const double beta = 1.0;

    // row-major GEMV
    // y = alpha * op(A) * x + beta * y
    //   = alpha * op(A) * x + y
    // for MxN matrix A
    // for vector x
    // for vector y

    // column-major cuBLAS GEMV for row-major stored A (and x,y)
    // for op: opposite of transpose_A

    oneapi::math::blas::column_major::gemv(
        queue,
        invert_transpose_operator(is_A_transposed),
        static_cast<std::int64_t>(N),
        static_cast<std::int64_t>(M),
        alpha_value,
        d_A,
        static_cast<std::int64_t>(N),
        d_x,
        1,
        beta,
        d_y,
        1);

    queue.wait();

    // return updated vector b
    return hpx::make_ready_future(d_y);
}

hpx::shared_future<double *>
ger(sycl::queue &queue,
    hpx::shared_future<double *> f_A,
    hpx::shared_future<double *> f_x,
    hpx::shared_future<double *> f_y,
    const std::size_t N)
{
    double *d_A = f_A.get();
    double *d_x = f_x.get();
    double *d_y = f_y.get();
    const double alpha = -1.0;

    // row-major GER
    // A = alpha * x*y^T + A
    //   = -x*y^T + A

    // column-major cuBLAS GER for row-major stored A (and x,y)
    // A = alpha * y*x^T + A
    //   = -y*x^T + A
    // for opposite order of x,y

    oneapi::math::blas::column_major::ger(queue, static_cast<std::int64_t>(N), static_cast<std::int64_t>(N), alpha, d_y, 1, d_x, 1, d_A, static_cast<std::int64_t>(N));

    queue.wait();

    // Return updated A
    return hpx::make_ready_future(d_A);
}

DotDiagSyrkKernel::DotDiagSyrkKernel(double *d_A, double *d_r, const std::size_t M, const std::size_t N):
d_A(d_A), d_r(d_r), M(M), N(N) {}

void DotDiagSyrkKernel::operator()(const sycl::id<1> &id) const
{
    double dot_product = 0.0;

    for (std::size_t i = 0; i < M; ++i)
    {
        dot_product += d_A[i * N + id] * d_A[i * N + id];
    }

    d_r[id] += dot_product;
}

hpx::shared_future<double *>
dot_diag_syrk(sycl::queue &queue,
              hpx::shared_future<double *> f_A,
              hpx::shared_future<double *> f_r,
              const std::size_t M,
              const std::size_t N)
{
    auto d_A = f_A.get();
    auto d_r = f_r.get();

    // r = r + diag(A^T * A)

    auto event = queue.submit
    (
        [&](sycl::handler &cgh)
        {
            auto kernel = DotDiagSyrkKernel(d_A, d_r, M, N);
            cgh.parallel_for(
                sycl::range<1>(N), kernel
            );
        }
    );
    event.wait();

    return hpx::make_ready_future(d_r);
}


DotDiagGemmKernel::DotDiagGemmKernel(double *A, double *B, double *r, const std::size_t M, const std::size_t N):
A(A), B(B), r(r), M(M), N(N) {}

void DotDiagGemmKernel::operator()(const sycl::id<1> &id) const
{
    double dot_product = 0.0;

    for (std::size_t i = 0; i < M; ++i)
    {
        dot_product += A[i * N + id] * B[id * M + i];
    }

    r[id] += dot_product;
}

hpx::shared_future<double *>
dot_diag_gemm(sycl::queue &queue,
              hpx::shared_future<double *> f_A,
              hpx::shared_future<double *> f_B,
              hpx::shared_future<double *> f_r,
              const std::size_t M,
              const std::size_t N)
{
    double *d_A = f_A.get();
    double *d_B = f_B.get();
    double *d_r = f_r.get();

    // r = r + diag(A * B)
    auto event = queue.submit
    (
        [&](sycl::handler &cgh)
        {
            auto kernel = DotDiagGemmKernel(d_A, d_B, d_r, M, N);
            cgh.parallel_for(
                sycl::range<1>(N), kernel
            );
        }
    );
    event.wait();

    return hpx::make_ready_future(d_r);
}

// BLAS LEVEL 1 OPERATIONS ////////////////////////////////////////////////////////////////////////////////////////////

hpx::shared_future<double *>
dot(
    sycl::queue &queue,
    hpx::shared_future<double *> f_a,
    hpx::shared_future<double *> f_b,
    const std::size_t N
)
{
    double *result = sycl::malloc_shared<gprat::sycl_backend::real_t>(1, queue);
    queue.fill(result, 0, 1).wait();

    double *d_a = f_a.get();
    double *d_b = f_b.get();

    oneapi::math::blas::column_major::dot(queue, static_cast<std::int64_t>(N), d_a, 1, d_b, 1, result);

    queue.wait();

    return hpx::make_ready_future(result);
}
