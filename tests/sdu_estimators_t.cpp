#include <catch2/catch.hpp>
#include <sdu_estimators/sdu_estimators.hpp>

using namespace sdu_estimators;

TEST_CASE("add_one", "[adder]")
{
  REQUIRE(add_one(0) == 1);
  REQUIRE(add_one(123) == 124);
  REQUIRE(add_one(-1) == 0);
}
