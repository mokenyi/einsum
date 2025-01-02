#include "get_combined_dims_view_test.hpp"
#include <array>
#include "xtensor/xtensor.hpp"
#include "xtensor/xarray.hpp"
#include "xtensor/xfixed.hpp"
#include <iostream>
#include "subscripts.hpp"
#include "has_repeated_labels.hpp"
#include "get_combined_dims_view.hpp"

void get_combined_dims_view_test() {
  std::array<size_t,2> sh2 = {2,2};
  xt::xtensor<double,2> x;

  xt::xtensor_fixed<int,xt::xshape<1,2,3>> y;

  xt::xarray<short> z = xt::zeros<short>({1,2,3,4});
  xt::xarray<short> w = xt::zeros<short>({1,2,3,4});

  using subs1 = subscripts<0, 1>;
  std::cout << has_repeated_labels<
    subs1::start_first,
    subs1::start_second,
    typename subs1::tuple_type
  >::value << std::endl;
    
  get_combined_dims_view<subs1>(x);

  using subs2 = subscripts<1, 1, 2, 3>;
  std::cout << has_repeated_labels<
    subs2::start_first,
    subs2::start_second,
    typename subs2::tuple_type
  >::value << std::endl;

  std::cout << "strides signed? " << std::is_signed<xt::xarray<short>::strides_type::value_type>::value << std::endl;
  std::cout << "shape signed? " << std::is_signed<xt::xarray<short>::shape_type::value_type>::value << std::endl;
  get_combined_dims_view<subs2>(std::move(w));
}
