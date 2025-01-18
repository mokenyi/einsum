#ifndef GET_COMBINED_DIMS_VIEW_HPP
#define GET_COMBINED_DIMS_VIEW_HPP
#include "xtensor/xtensor.hpp"
#include "xtensor/xstrided_view.hpp"
#include "has_repeated_labels.hpp"
#include <iostream>

template<typename I, typename E>
xt::xstrided_view<
  xt::xclosure_t<E>,
  xt::svector<size_t>,
  xt::layout_type::dynamic
> get_combined_dims_view(E&& x, int ellipsis) {
  auto& y = x;
  int const ndim = y.dimension();
  std::vector<int> const labels = I::parse_operand_subscripts(ndim, ellipsis);
  std::vector<int> icombinemap(ndim, -1);

  int icombine(0);
  std::vector<size_t> new_dims;
  std::vector<long> new_strides;

  for (int idim = 0; idim < ndim; ++idim) {
    int const label = labels.at(idim);
    int const dim = y.shape(idim);
    int const stride = y.strides()[idim];

    if (label >= 0) {
      icombinemap.at(idim) = icombine;
      new_dims.push_back(dim);
      new_strides.push_back(stride);
      ++icombine;
    }
    else {
      int const i = icombinemap.at(idim + label);
      icombinemap.at(idim) = -1;
      if (new_dims.at(i) != dim) {
        std::stringstream ss;
        ss << "Cannot collapse dimension " << idim << " (of size ";
        ss << dim << ") into dimension " << idim + label << " (of size ";
        ss << new_dims.at(i) << ") for subscripts " << I::to_string();
        throw std::invalid_argument(ss.str());
      }
      new_strides.at(i) += stride;
    }
  }

  using shape_type = xt::svector<size_t>;
  using strides_type = xt::get_strides_t<shape_type>;

  return xt::strided_view(y,shape_type{new_dims.begin(), new_dims.end()},strides_type{new_strides.begin(), new_strides.end()},0LU,xt::layout_type::dynamic);
}

#endif // GET_COMBINED_DIMS_VIEW_HPP
