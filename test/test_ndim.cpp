#include "doctest.h"
#include "xtensor/xtensor.hpp"
#include "xtensor/xarray.hpp"
#include "xtensor/xfixed.hpp"
#include "einsum.hpp"

TEST_CASE("ndim should return the numbers of dimensions of a set of "
  "operands") {
  xt::xtensor<short,1> x;
  xt::xtensor_fixed<int,xt::xshape<1,2,3>> y;
  xt::xarray<long> z = xt::zeros<long>({1,2,3,4});

  auto const t = std::make_tuple(x,y,z);
  std::array<size_t,3> actual;
  std::array<size_t,3> expected = {1,3,4};

  get_ndim(t,actual);
  CHECK(actual.at(0) == expected.at(0));
  CHECK(actual.at(1) == expected.at(1));
  CHECK(actual.at(2) == expected.at(2));
}
