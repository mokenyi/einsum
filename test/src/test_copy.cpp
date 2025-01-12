#include "doctest.h"
#include <tuple>
#include <array>
#include "einsum.hpp"

TEST_CASE("Copy the contents of a tuple into an array") {
  auto t = std::make_tuple(42, 1993, 0);
  std::array<int,3> const expected = {42, 1993, 0};
  std::array<int,3> actual;

  copy(actual, t);
  CHECK(actual == expected);
}

