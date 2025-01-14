#include "doctest.h"
#include <array>
#include "xtensor/xtensor.hpp"
#include "xtensor/xfixed.hpp"
#include "xtensor/xarray.hpp"
#include "operands_container.hpp"
#include "xtensor/xio.hpp"
#include "test_util.hpp"

std::vector<std::tuple<double,int,short>> get_test_data() {
  std::vector<std::tuple<double,int,short>> ret;
  ret.push_back(std::make_tuple(0.0,0,0));
  ret.push_back(std::make_tuple(0.0,4,120));
  ret.push_back(std::make_tuple(1.0,1,10));
  ret.push_back(std::make_tuple(1.0,5,130));
  ret.push_back(std::make_tuple(2.0,2,20));
  ret.push_back(std::make_tuple(2.0,6,140));
  ret.push_back(std::make_tuple(3.0,3,30));
  ret.push_back(std::make_tuple(3.0,7,150));
  ret.push_back(std::make_tuple(0.0,8,40));
  ret.push_back(std::make_tuple(0.0,12,160));
  ret.push_back(std::make_tuple(1.0,9,50));
  ret.push_back(std::make_tuple(1.0,13,170));
  ret.push_back(std::make_tuple(2.0,10,60));
  ret.push_back(std::make_tuple(2.0,14,180));
  ret.push_back(std::make_tuple(3.0,11,70));
  ret.push_back(std::make_tuple(3.0,15,190));
  ret.push_back(std::make_tuple(0.0,16,80));
  ret.push_back(std::make_tuple(0.0,20,200));
  ret.push_back(std::make_tuple(1.0,17,90));
  ret.push_back(std::make_tuple(1.0,21,210));
  ret.push_back(std::make_tuple(2.0,18,100));
  ret.push_back(std::make_tuple(2.0,22,220));
  ret.push_back(std::make_tuple(3.0,19,110));
  ret.push_back(std::make_tuple(3.0,23,230));

  return ret;
}

TEST_CASE("Iterator iterates a set of operands in the correct order") {
  std::array<size_t,2> sh2 = {1,4};
  xt::xtensor<double,2> x = xt::arange(0, 4).reshape({1,4});
  typedef xt::xshape<3,2,4> sh3;
  xt::xtensor_fixed<int,sh3> y = xt::arange(0, 24).reshape({3,2,4});

  xt::xarray<short> z = xt::arange(0,240,10).reshape({1,2,3,4});

  auto t = std::make_tuple(x, y, z);
  // shape of iterator should be [3, 1, 4, 2]

  std::array<std::vector<int>,3> op_axes;
  op_axes.at(0) = {-1,  0,  1, -1};
  op_axes.at(1) = { 0, -1,  2,  1},
  op_axes.at(2) = { 2,  0,  3,  1};

  auto container = make_operands_container(t, op_axes);
  auto it = container.begin();

  auto const expected = get_test_data();

  for (auto expected_it = expected.cbegin(); expected_it != expected.cend(); ++expected_it) {
    CHECK(equal_to_pointees(*it++,*expected_it));
  }

  CHECK(it == container.end());
}

