#include "doctest.h"
#include "einsum.hpp"
#include "xtensor/xio.hpp"
#include "xtensor/xtensor.hpp"
#include "xtensor/xfixed.hpp"
#include "labels.hpp"
#include "subscripts.hpp"

TEST_CASE("einsum::eval() evaluates an Einstein summation of two operands") {
  xt::xtensor<double,2> x = {
    {0.82450064, 0.58973638, 0.26651105, 0.33222087},
    {0.2477022 , 0.93260011, 0.29248315, 0.04318276}
  };
  xt::xtensor_fixed<double,xt::xshape<3,4>> y = {
    {0.11357894, 0.66455996, 0.75776223, 0.74428668},
    {0.12294283, 0.26457651, 0.93717736, 0.68914559},
    {0.81280101, 0.43663051, 0.88750341, 0.32555251}
  };

  xt::xarray<double> const z = einsum<subscripts<I,J>,subscripts<K,J>>(_)
    .eval<subscripts<K,I>>(x, y);

  std::cout << z << std::endl;
} 

