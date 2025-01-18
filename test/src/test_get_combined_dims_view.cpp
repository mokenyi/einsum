#include "doctest.h"
#include "einsum.hpp"
#include "xtensor/xio.hpp"
#include "xtensor/xtensor.hpp"
#include "xtensor/xfixed.hpp"
#include "labels.hpp"
#include "subscripts.hpp"

TEST_CASE("get_combined_dims_view should not combine dimensions if label") {
}

TEST_CASE("get_combined_dims_view should combine dimensions if labels are "
  "repeated") {
  std::array<size_t,2> sh2 = {25,25};
  xt::xtensor<double,2> x = xt::arange<double>(25).reshape({5,5});
  using subs0 = subscripts<I,I>;
  auto const actualx = get_combined_dims_view<subs0>(x, _);

  xt::xtensor<double,1> const expectedx = {0., 6., 12., 18., 24.};
  CHECK(expectedx == actualx);

  xt::xarray<short> w = xt::arange<short>(48).reshape({2,2,3,4});
  using subs1 = subscripts<I,I,J,K>;
  auto const actualw = get_combined_dims_view<subs1>(w, _);

  xt::xtensor<short,3> const expectedw = {
    {{ 0,  1,  2,  3},
     { 4,  5,  6,  7},
     { 8,  9, 10, 11}},
    {{36, 37, 38, 39},
     {40, 41, 42, 43},
     {44, 45, 46, 47}}
  };

  CHECK(expectedw == actualw);
}
