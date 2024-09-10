#include <sdu_estimators/regressor_extensions/kreisselmeier.hpp>

namespace sdu_estimators::regressor_extensions
{
  Kreisselmeier::~Kreisselmeier() = default;

  Kreisselmeier::Kreisselmeier(float dt, float ell)
  {
    this->dt = dt;
    this->ell = ell;

    first_run = true;
  }

  void Kreisselmeier::step(const Eigen::VectorXd &y, const Eigen::MatrixXd &phi)
  {
    if (first_run)
    {
      // int n = y.size();
      int m = phi.rows();

      y_f = Eigen::VectorXd(m, 1);
      phi_f = Eigen::MatrixXd(m, m);

      first_run = false;
    }

    phi_f += dt * (
          -ell * phi_f + phi * phi.transpose()
        );

    y_f += dt * (
      -ell * y_f + phi * y
    );
  }

  Eigen::VectorXd Kreisselmeier::getY()
  {
    return y_f;
  }

  Eigen::MatrixXd Kreisselmeier::getPhi()
  {
    return phi_f;
  }

  void Kreisselmeier::reset()
  {
    first_run = true;
    y_f *= 0;
    phi_f *= 0;
  }


}