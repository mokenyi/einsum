#include "multi_iterator.hpp"
#include "xtensor/xtensor.hpp"
#include "xtensor/xarray.hpp"
#include "xtensor/xfixed.hpp"

int main(int argc, char *argv[]) {
  std::array<size_t,2> sh2 = {2,2};
  xt::xtensor<double,2> x(sh2);

  xt::xtensor_fixed<int,xt::xshape<1,2,3>> y;

  xt::xarray<short> z = xt::zeros<short>({1,2,3,4});
  xt::xarray<short> w = xt::zeros<short>({1,2,3,4});

  auto t = std::make_tuple(x, y, z);
  xt::xtensor<size_t,2> op_axes;
  auto mi = make_multi_iterator(t, op_axes);

  return 0;
}
