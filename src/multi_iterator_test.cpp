#include "multi_iterator_test.hpp"

#include "multi_iterator.hpp"
#include "xtensor/xtensor.hpp"
#include "xtensor/xarray.hpp"
#include "xtensor/xfixed.hpp"
#include "xtensor/xio.hpp"


void multi_iterator_test() {
  std::array<size_t,2> sh2 = {1,4};
  xt::xtensor<double,2> x = xt::arange(0, 4).reshape({1,4});
  std::cout << x << std::endl;
  typedef xt::xshape<3,2,4> sh3;
  xt::xtensor_fixed<int,sh3> y = xt::arange(0, 24).reshape({3,2,4});

  xt::xarray<short> z = xt::arange(0,240,10).reshape({1,2,3,4});

  auto t = std::make_tuple(x, y, z);
  // shape of iterator should be [3, 1, 4, 2]

  std::array<std::vector<int>,3> op_axes;
  op_axes.at(0) = {-1,  0,  1, -1};
  op_axes.at(1) = { 0, -1,  2,  1},
  op_axes.at(2) = { 2,  0,  3,  1};
  for (int i=0; i<z.size(); ++i) {
    std::cout << " " << z.data() + i << "," << *(z.data() + i);
  }
  std::cout << std::endl;
  auto mi = make_multi_iterator(t, op_axes);
  mi.print_current();
  while (true) {
    mi.next();
    mi.print_current();
  }
}
