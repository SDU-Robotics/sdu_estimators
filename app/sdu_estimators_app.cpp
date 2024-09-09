#include "sdu_estimators/sdu_estimators.hpp"
#include <iostream>

int
main()
{
  int result = sdu_estimators::add_one(1);
  std::cout << "1 + 1 = " << result << std::endl;
}
