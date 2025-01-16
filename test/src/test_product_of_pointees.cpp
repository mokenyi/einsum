#include "doctest.h"
#include "einsum_util.hpp"

TEST_CASE("product_of_pointees should get the product of the pointees of the "
  "first N pointers in a tuple of pointers") {
  int const i = 42;
  double const d = 3.14;
  double const f = 2.718;
  short const s = 1993;
  double const u = 1e10;

  auto t = std::make_tuple(&i, &d, &f, &s, &u);
  
  double const expected = 714390.53112;

  double const actual = product_of_pointees<0,4>(t);

  CHECK(actual == expected);
}
