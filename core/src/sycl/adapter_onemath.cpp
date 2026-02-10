#include "sycl/adapter_onemath.hpp"

// BLAS LEVEL 3 OPERATIONS ////////////////////////////////////////////////////////////////////////////////////////////

double *
potrf(sycl::queue queue, double *f_A, const std::size_t N)
{
    std::cout << "[adapter_onemath.cpp] [potrf] : Entering \n";

    auto ptype_A = sycl::get_pointer_type(f_A, queue.get_context());

    std::cout << "[adapter_onemath.cpp] [potrf] : Pointer types - A: " 
    << usm_alloc_to_string(ptype_A) << "\n";

    std::int64_t scratchpad_size = oneapi::math::lapack::potrf_scratchpad_size<double>(
        queue, 
        oneapi::math::uplo::upper, 
        static_cast<std::int64_t>(N), 
        static_cast<std::int64_t>(N)
    );

    gprat::sycl_backend::real_t *scratchpad = sycl::malloc_device<gprat::sycl_backend::real_t>(static_cast<std::size_t>(scratchpad_size), queue);

    // row-major POTRF
    // A = potrf(A)
    // for LOWER part of symmetric positive semi-definite matrix A

    // column-major cuBLAS POTRF for row-major stored A
    // for UPPER part of symmetric positive semi-definite matrix A

    queue.wait();

    oneapi::math::lapack::potrf(queue, oneapi::math::uplo::upper, static_cast<std::int64_t>(N), f_A, static_cast<std::int64_t>(N), scratchpad, scratchpad_size);
    
    queue.wait();

    sycl::free(scratchpad, queue);

    std::cout << "[adapter_onemath.cpp] [potrf] : Leaving \n";

    return f_A;
}

double *
trsm(
     sycl::queue queue,
     double *f_A,
     double *f_B,
     const std::size_t M,
     const std::size_t N,
     const oneapi::math::transpose is_transposed,
     const oneapi::math::side is_right)
{
    std::cout << "[adapter_onemath.cpp] [trsm] : Entering \n";
    // TRSM constants
    const double alpha = 1.0;

    auto ptype_A = sycl::get_pointer_type(f_A, queue.get_context());
    auto ptype_B = sycl::get_pointer_type(f_B, queue.get_context());

    std::cout << "[adapter_onemath.cpp] [trsm] : Pointer types - A: " 
    << usm_alloc_to_string(ptype_A) << ", B: " 
    << usm_alloc_to_string(ptype_B) << "\n";

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

    queue.wait();

    oneapi::math::blas::column_major::trsm(
        queue,
        invert_side_operator(is_right),
        oneapi::math::uplo::upper,
        is_transposed,
        oneapi::math::diag::nonunit,
        static_cast<std::int64_t>(M),
        static_cast<std::int64_t>(N),
        alpha,
        f_A,
        static_cast<std::int64_t>(M),
        f_B,
        static_cast<std::int64_t>(N));

    queue.wait();
    std::cout << "[adapter_onemath.cpp] [trsm] : Leaving \n";
    
    return f_B;
}

double *
syrk(sycl::queue queue,
     double *f_A,
     double *f_C,
     const std::size_t N)
{
    std::cout << "[adapter_onemath.cpp] [syrk] : Entering \n";
    // SYRK constants
    const double alpha = -1.0;
    const double beta = 1.0;

    auto ptype_A = sycl::get_pointer_type(f_A, queue.get_context());
    auto ptype_C = sycl::get_pointer_type(f_C, queue.get_context());

    std::cout << "[adapter_onemath.cpp] [syrk] : Pointer types - A: " 
    << usm_alloc_to_string(ptype_A) << ", C: " 
    << usm_alloc_to_string(ptype_C) << "\n";

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

    queue.wait();

    oneapi::math::blas::column_major::syrk(
        queue,
        oneapi::math::uplo::upper,
        oneapi::math::transpose::trans,
        static_cast<std::int64_t>(N),
        static_cast<std::int64_t>(N),
        alpha,
        f_A,
        static_cast<std::int64_t>(N),
        beta,
        f_C,
        static_cast<std::int64_t>(N));

    queue.wait();

    std::cout << "[adapter_onemath.cpp] [syrk] : Leaving \n";

    return f_C;
}

double *
gemm(sycl::queue queue,
     double *f_A,
     double *f_B,
     double *f_C,
     const std::size_t M,
     const std::size_t N,
     const std::size_t K,
     const oneapi::math::transpose is_A_transposed,
     const oneapi::math::transpose is_B_transposed)
{
    std::cout << "[adapter_onemath.cpp] [gemm] : Entering \n";

    const double alpha = -1.0;
    const double beta = 1.0;

    // row-major GEMM
    // C = alpha * op(A) * op(B) + beta * C
    //   = op(A) * op(B) - C
    // for op(A): transpose_A
    // for op(B): transpose_B

    // column-major cuBLAS GEMM for row-major stored A, B, C
    // C = alpha * op(B) * op(A) + beta * C
    //   = op(B) * op(A) - C
    // for inverted ordering of matrices A, B

    auto ptype_A = sycl::get_pointer_type(f_A, queue.get_context());
    auto ptype_B = sycl::get_pointer_type(f_B, queue.get_context());
    auto ptype_C = sycl::get_pointer_type(f_C, queue.get_context());

    std::cout << "[adapter_onemath.cpp] [gemm] : Pointer types - A: " 
    << usm_alloc_to_string(ptype_A) << ", B: " 
    << usm_alloc_to_string(ptype_B) << ", C: " 
    << usm_alloc_to_string(ptype_C) << "\n";

    queue.wait();

    oneapi::math::blas::column_major::gemm(
        queue,
        is_B_transposed,
        is_A_transposed,
        static_cast<std::int64_t>(N),
        static_cast<std::int64_t>(M),
        static_cast<std::int64_t>(K),
        alpha,
        f_B,
        static_cast<std::int64_t>(N),
        f_A,
        static_cast<std::int64_t>(K),
        beta,
        f_C,
        static_cast<std::int64_t>(N)); 

    queue.wait();

    std::cout << "[adapter_onemath.cpp] [gemm] : Leaving \n";

    return f_C;
}

// BLAS LEVEL 2 OPERATIONS ////////////////////////////////////////////////////////////////////////////////////////////

double *
trsv(sycl::queue queue,
     double *f_A,
     double *f_b,
     const std::size_t N,
     const oneapi::math::transpose is_A_transposed)
{
    std::cout << "[adapter_onemath.cpp] [trsv] : Entering \n";
    auto ptype_A = sycl::get_pointer_type(f_A, queue.get_context());
    auto ptype_B = sycl::get_pointer_type(f_b, queue.get_context());

    std::cout << "[adapter_onemath.cpp] [trsv] : Pointer types - A: " 
    << usm_alloc_to_string(ptype_A) << ", b: " 
    << usm_alloc_to_string(ptype_B) << "\n";

    // row-major TRSV solves for x
    // op(A) * x = b
    // for op: transpose_A
    // for LOWER part of lower triangular matrix A

    // column-major cuBLAS TRSV for row-major stored A
    // for op: opposite of transpose_A
    // for UPPER part of lower triangular matrix A

    queue.wait();

    oneapi::math::blas::column_major::trsv(
        queue,
        oneapi::math::uplo::upper,
        invert_transpose_operator(is_A_transposed),
        oneapi::math::diag::nonunit,
        static_cast<std::int64_t>(N),
        f_A,
        static_cast<std::int64_t>(N),
        f_b,
        1);

    queue.wait();
    std::cout << "[adapter_onemath.cpp] [trsv] : Leaving \n";
    // return solution vector x
    return f_b;
}

double *
gemv(sycl::queue queue,
     double *f_A,
     double *f_x,
     double *f_y,
     const std::size_t M,
     const std::size_t N,
     const double alpha,
     const oneapi::math::transpose is_A_transposed)
{
    std::cout << "[adapter_onemath.cpp] [gemv] : Entering \n";
    const double alpha_value = alpha;
    const double beta = 1.0;

    auto ptype_A = sycl::get_pointer_type(f_A, queue.get_context());
    auto ptype_B = sycl::get_pointer_type(f_x, queue.get_context());
    auto ptype_C = sycl::get_pointer_type(f_y, queue.get_context());

    std::cout << "[adapter_onemath.cpp] [gemv] : Pointer types - A: " 
    << usm_alloc_to_string(ptype_A) << ", x: " 
    << usm_alloc_to_string(ptype_B) << ", y: " 
    << usm_alloc_to_string(ptype_C) << "\n";

    // row-major GEMV
    // y = alpha * op(A) * x + beta * y
    //   = alpha * op(A) * x + y
    // for MxN matrix A
    // for vector x
    // for vector y

    // column-major cuBLAS GEMV for row-major stored A (and x,y)
    // for op: opposite of transpose_A

    queue.wait();

    oneapi::math::blas::column_major::gemv(
        queue,
        invert_transpose_operator(is_A_transposed),
        static_cast<std::int64_t>(N),
        static_cast<std::int64_t>(M),
        alpha_value,
        f_A,
        static_cast<std::int64_t>(N),
        f_x,
        1,
        beta,
        f_y,
        1);

    queue.wait();

    std::cout << "[adapter_onemath.cpp] [gemv] : Leaving \n";

    // return updated vector b
    return f_y;
}

double *
ger(sycl::queue queue,
    double *f_A,
    double *f_x,
    double *f_y,
    const std::size_t N)
{
    std::cout << "[adapter_onemath.cpp] [ger] : Entering \n";
    const double alpha = -1.0;

    auto ptype_A = sycl::get_pointer_type(f_A, queue.get_context());
    auto ptype_B = sycl::get_pointer_type(f_x, queue.get_context());
    auto ptype_C = sycl::get_pointer_type(f_y, queue.get_context());

    std::cout << "[adapter_onemath.cpp] [ger] : Pointer types - A: " 
    << usm_alloc_to_string(ptype_A) << ", x: " 
    << usm_alloc_to_string(ptype_B) << ", y: " 
    << usm_alloc_to_string(ptype_C) << "\n";

    // row-major GER
    // A = alpha * x*y^T + A
    //   = -x*y^T + A

    // column-major cuBLAS GER for row-major stored A (and x,y)
    // A = alpha * y*x^T + A
    //   = -y*x^T + A
    // for opposite order of x,y

    queue.wait();

    oneapi::math::blas::column_major::ger(queue, static_cast<std::int64_t>(N), static_cast<std::int64_t>(N), alpha, f_y, 1, f_x, 1, f_A, static_cast<std::int64_t>(N));

    queue.wait();

    std::cout << "[adapter_onemath.cpp] [ger] : Leaving \n";
    return f_A;
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

double *
dot_diag_syrk(sycl::queue queue,
              double *f_A,
              double *f_r,
              const std::size_t M,
              const std::size_t N)
{
    // r = r + diag(A^T * A)

    auto event = queue.submit
    (
        [&](sycl::handler &cgh)
        {
            auto kernel = DotDiagSyrkKernel(f_A, f_r, M, N);
            cgh.parallel_for(
                sycl::range<1>(N), kernel
            );
        }
    );
    event.wait();

    return f_r;
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

double *
dot_diag_gemm(sycl::queue queue,
              double *f_A,
              double *f_B,
              double *f_r,
              const std::size_t M,
              const std::size_t N)
{
    // r = r + diag(A * B)
    auto event = queue.submit
    (
        [&](sycl::handler &cgh)
        {
            auto kernel = DotDiagGemmKernel(f_A, f_B, f_r, M, N);
            cgh.parallel_for(
                sycl::range<1>(N), kernel
            );
        }
    );
    event.wait();

    return f_r;
}

// BLAS LEVEL 1 OPERATIONS ////////////////////////////////////////////////////////////////////////////////////////////

double *
dot(
    sycl::queue queue,
    double *f_a,
    double *f_b,
    const std::size_t N
)
{
    double *result = sycl::malloc_device<gprat::sycl_backend::real_t>(1, queue);
    queue.fill(result, 0, 1).wait();

    oneapi::math::blas::column_major::dot(queue, static_cast<std::int64_t>(N), f_a, 1, f_b, 1, result);

    queue.wait();

    return result;
}
