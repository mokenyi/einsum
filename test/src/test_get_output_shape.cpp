#include "doctest.h"
#include "einsum_util.hpp"

TEST_CASE("Should get correct shape of einsum output") {
  std::array<std::vector<int>,4> op_axes;
  op_axes.at(0) = {-1, -1, -1, -1, 0, 1, 2, -1};
  op_axes.at(1) = {-1, -1, -1, 1, -1, 0, 3, 2};
  op_axes.at(2) = {4, 2, 3, -1, 1, -1, -1, 0};
  op_axes.at(3) = {0, 1, 2, 3, -1, -1, -1, -1};

  std::array<std::vector<size_t>,3> op_sh;
  op_sh.at(0) = {2, 3, 4};
  op_sh.at(1) = {3, 5, 7, 4};
  op_sh.at(2) = {6, 2, 8, 1, 9};

  std::vector<size_t> const expected = get_output_shape(op_axes, op_sh);
  std::vector<size_t> const actual = {9, 8, 1, 5};

  CHECK(expected == actual);
}

