//#include "einsum.hpp"
#include "xtensor/xtensor.hpp"
#include "xtensor/xfixed.hpp"
#include "xtensor/xarray.hpp"
#include "xtensor/xstrided_view.hpp"
#include "xtensor/xio.hpp"
#include "xtensor/xlayout.hpp"
#include "types_user.hpp"
#include "types_wrapper.hpp"
#include "einsum.hpp"
#include "einsum_test.hpp"

void layout_test() {
  xt::xtensor_fixed<int,xt::xshape<4,3,3>,xt::layout_type::column_major> x = xt::arange(36).reshape({4,3,3});
  std::array<long,2> sh = {4,3};
  std::array<long,2> st = {1,16};
  
  auto v = xt::strided_view(x, std::move(sh), std::move(st), 0LU, xt::layout_type::column_major);
  std::cout << v << std::endl;
}

int main(int argc, char *argv[]) {
  einsum_test();
  return 0;
}
