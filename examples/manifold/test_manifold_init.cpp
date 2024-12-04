#include "sdu_estimators/math/riemannian_manifolds/sphere.hpp"
#include "sdu_estimators/math/riemannian_manifolds/symmetric_positive_definite.hpp"

void test_sphere()
{
  sdu_estimators::math::manifold::Sphere<double, 3> sphere;

  Eigen::Vector3d v1, v2;
  v1.setRandom().normalize();
  v2.setRandom().normalize();

  std::cout << "v1 " << v1 << std::endl;
  std::cout << "v2 " << v2 << std::endl;

  // double d = sphere.dist(v1, v2);
  // std::cout << "dist " << d << std::endl;

  Eigen::Vector3d v3 = v2 * 2;

  auto projv = sphere.projection(v1, v3);
  std::cout << "projv\n" << projv << std::endl;

  auto retr = sphere.retraction(v1, v3);
  std::cout << "retr\n" << retr << std::endl;


  auto expv = sphere.exp(v1, v3);
  std::cout << "expv\n" << expv << std::endl;

  auto logv = sphere.log(v1, v3);
  std::cout << "logv\n" << logv << std::endl;
}

void test_spd()
{
  sdu_estimators::math::manifold::SymmetricPositiveDefinite<double, 3> spd;

  Eigen::Matrix3d m1, m2, eta;
  m1 << 1, 0, 0,
        0, 2, 0,
        0, 0, 3;
  m2 << 5, 0, 0,
        0, 2, 0,
        0, 0, 3;

  eta <<    0.3188,    0.3426,   -1.3499,
   -1.3077,    3.5784,    3.0349,
   -0.4336,   2.7694,    0.7254;

  double normv = spd.norm(m1, eta);
  std::cout << "normv\n" << normv << std::endl;

  double d = spd.dist(m1, m2);
  std::cout << "d\n" << d << std::endl;

  auto retr = spd.retraction(m1, eta);
  std::cout << "retr\n" << retr << std::endl;

  auto exp = spd.exp(m1, eta);
  std::cout << "exp\n" << exp << std::endl;

  auto log = spd.log(m1, eta);
  std::cout << "log\n" << log << std::endl;
}

int main()
{
  // test_sphere();

  //
  test_spd();

  return 0;
}