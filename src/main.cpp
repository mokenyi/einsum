#include "einsum.hpp"
#include "xtensor/xtensor.hpp"
#include "xtensor/xfixed.hpp"
#include "xtensor/xarray.hpp"

int main(int argc, char *argv[]) {
  std::array<size_t,2> sh2 = {1,4};
  xt::xtensor<double,2> x = xt::arange(0, 4).reshape({1,4});
  typedef xt::xshape<3,2,4> sh3;
  xt::xtensor_fixed<int,sh3> y = xt::arange(0, 24).reshape({3,2,4});

  xt::xarray<short> z = xt::arange(0,240,10).reshape({1,2,3,4});
  einsum("", x, y, z);
  return 0;
}
