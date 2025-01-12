#include "doctest.h"
#include <tuple>
#include <array>
#include "einsum.hpp"
#include "xtensor/xtensor.hpp"
#include "xtensor/xfixed.hpp"
#include "xtensor/xarray.hpp"
#include "labels.hpp"

TEST_CASE("Return value should map broadcast dims in the output to unlabelled "
  "and inserted dims in the inputs") {
  std::array<size_t,2> sh2 = {3,6};
  xt::xtensor<short,2> x(sh2);
  using xfixed_type = xt::xtensor_fixed<int,xt::xshape<3,3,5,3>>;
  using xfixed_shape_type = xfixed_type::inner_shape_type;
  xfixed_shape_type ignored{ {0} };
  xfixed_type const y(
      ignored,
      0
  );
  xt::xarray<long> const z = xt::zeros<long>({3,4,3});
  
  auto ops = std::make_tuple(y,z,x);
  std::array<std::vector<int>,3> combined_labels;
  combined_labels.at(0) = std::vector<int>({I,J,_,_});
  combined_labels.at(1) = std::vector<int>({J,K,_});
  combined_labels.at(2) = std::vector<int>({I,L});

  std::vector<int> const iter_labels({L,K,_,_,I,J});
  int const ellipsis(_);

  std::array<std::vector<int>,4> actual = prepare_op_axes_loop(
    combined_labels,
    iter_labels,
    ellipsis
  );

  std::array<std::vector<int>,4> expected;
  expected.at(0) = {-1, -1,  2,  3,  0,  1};
  expected.at(1) = {-1,  1, -1,  2, -1,  0};
  expected.at(2) = { 1, -1, -1, -1,  0, -1};

  CHECK(actual.at(0) == expected.at(0));
  CHECK(actual.at(1) == expected.at(1));
  CHECK(actual.at(2) == expected.at(2));
}

