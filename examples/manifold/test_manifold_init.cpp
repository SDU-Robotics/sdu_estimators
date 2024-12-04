#include "sdu_estimators/math/riemannian_manifolds/sphere.hpp"

int main()
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

  return 0;
}