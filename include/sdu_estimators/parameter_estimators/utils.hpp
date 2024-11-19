#pragma once
#ifndef PARAMETER_ESTIMATORS_UTILS_HPP
#define PARAMETER_ESTIMATORS_UTILS_HPP

#include <Eigen/Dense>


namespace sdu_estimators::parameter_estimators::utils
{
    // https://gist.github.com/redpony/fc8a0db6b20f7b1a3f23

    // set use_cholesky if M is symmetric - it's faster and more stable
    // for dep paring it won't be
    template <typename MatrixType>
    inline typename MatrixType::Scalar logdet(const MatrixType& M, bool use_cholesky = false) {
        using namespace Eigen;
        using std::log;
        typedef typename MatrixType::Scalar Scalar;
        Scalar ld = 0;
        if (use_cholesky) {
            LLT<Matrix<Scalar,Dynamic,Dynamic>> chol(M);
            auto& U = chol.matrixL();
            for (unsigned i = 0; i < M.rows(); ++i)
                ld += log(U(i,i));
            ld *= 2;
        } else {
            PartialPivLU<Matrix<Scalar,Dynamic,Dynamic>> lu(M);
            auto& LU = lu.matrixLU();
            Scalar c = lu.permutationP().determinant(); // -1 or 1
            for (unsigned i = 0; i < LU.rows(); ++i) {
                const auto& lii = LU(i,i);
                if (lii < Scalar(0)) c *= -1;
                ld += log(abs(lii));
            }
            ld += log(c);
        }
        return ld;
    }

    enum class IntegrationMethod
    {
      Euler,
      Heuns
    };
}

#endif
