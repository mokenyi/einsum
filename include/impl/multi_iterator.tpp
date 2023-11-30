#include "util.hpp"

template<typename... Ts>
multi_iterator::multi_iterator(Ts... x, xt::xtensor<size_t, ndim> op_axes)
: exp(std::tuple::make_tuple(x))
, op_axes(op_axes)
, ndim(op_axes.shape(1))
{
  idx.resize(ndim);
  std::fill(idx.begin(), idx.end(), 0);

  xt::xtensor<size_t> xt_shape = xt::amax(op_axes, 0);
  // Each dim in the operator must map to a non-broadcast dim of at least one
  // operand.
  assert(xt::all(xt_shape != -1));

  shape.resize(ndim);
  std::copy(xt_shape.begin(), xt_shape.end(), shape.begin());
}

void multi_iterator::next()
{
  if (!hasnext()) {
    throw std::out_of_range("multi_iterator out of range!");
  }

  for (int i=static_cast<int>(ndim-1); i>=0; --i)
  {
    if (idx.at(i) < shape.at(i)-1)
    {
      ++idx.at(i);
      break;
    }
    else
    {
      idx.at(i) = 0;
    }
  }

  mtoo::zip(exp, current,
      [&idx, &op_axes](const auto& x, auto& y, size_t arg_idx)
      {
        size_t const ndim = x.dimension();
        size_t el_idx = std::vector<size_t>(ndim);
        for (int j=ndim-1; j>=0; --j) {
          // TODO: This isn't legal, but is conceptually correct.
          el_idx.at(j) = idx.at(
              std::find(op_axes[arg_idx].begin(), op_axes[arg_idx].end(), j) -
                op_axes[arg_idx].begin()
              );
        }

        size_t const flat_idx = flatten(el_idx);
        y = &*(x.begin() + flat_idx);
      }
}
