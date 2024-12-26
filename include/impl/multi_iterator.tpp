#include "util.hpp"

template<typename... Ts>
multi_iterator::multi_iterator(
  std::tuple<Ts...> const& x,
  xt::xtensor<size_t,2> const& op_axes
)
: exp(x)
, op_axes(op_axes)
, ndim(op_axes.shape(1))
, idx(0LU);
{
  op_idx.fill(0);
  std::array<size_t,2> op_sh_sh;
  op_sh_sh.at(0) = num_ops;
  op_sh_sh.at(1) = mtoo::get_max_op_ndim(x);
  op_sh.resize(op_sh_sh);
  op_sh = -1;

  set_op_sh(x);
  set_shape();
  init_current<num_ops-1>();
}

template<size_t N>
void init_current() {
  std::get<N>(current) = std::get<N>(x).data();
  init_current<N-1>();
}

template<>
void init_current<0>() {
  std::get<0LU>(current) = std::get<0LU>(x).data();
}

void multi_iterator::set_shape() {
  std::array<size_t,num_op> i;
  std::iota(i.begin(), i.end(), 0);
  shape.resize(ndim);
  for (size_t j=0; j<ndim; ++j) {
    // Points to first operand which has a non-broadcast axis that maps
    // to dim j of the iterator.
    auto non_broadcast_op_it = std::find_if(
      i.begin(),
      i.end(),
      [&op_axes, j](size_t i_p) {
        op_axes.at(i_p, j) != -1;
      }
    );
    size_t const size_dim_j = op_sh.at(
      *non_broadcast_op_it,
      op_axes.at(*non_broadcast_op_it, j)
    );
    auto bad_op_it = std::find_if_not(i.begin(), i.end(),
      [&op_axes, j, size_dim_j](size_t i_p) {
        return op_axes.at(i_p, j) == -1 || op_sh.at(i_p, op_axes.at(i_p, j)) == size_dim_j;
      });

    if (bad_op_it != i.end()) {
      std::stringstream ss;
      size_t const bad_axis = op_axes.at(*bad_op, j);
      ss << "Axis " << bad_axis " of operand " << *bad_op_it;
      ss << " has an incorrect length " << op_sh.at(*bad_op, bad_axis);
      ss << " != " << size_dim_j;
      throw std::invalid_argument(ss.str());
    }

    shape.at(j) = size_dim_j;
  }
}

template<size_t I>
void multi_iterator::bump_index(int j) {
  auto op = std::get<I>();
  if (op_axes.at(I, j) != -1) {
    current.get<I>() += op.strides(j);
  }
  set_current<I-1>();
}

template<>
void multi_iterator::bump_index<0>(int j) {
  auto op = std::get<I>();
  if (op_axes.at(0, j) != -1) {
    current.get<0>() += op.strides(j);
  }
}

void multi_iterator::next()
{
  if (!hasnext()) {
    throw std::out_of_range("multi_iterator out of range!");
  }

  for (int j=static_cast<int>(ndim-1); j>=0; --j)
  {
    if (idx.at(j) < shape.at(j)-1)
    {
      ++idx.at(j);
      bump_index<num_ops-1>();
      break;
    }
    else
    {
      idx.at(j) = 0;
    }
  }

//  mtoo::zip(exp, current,
//      [&idx, &op_axes](const auto& x, auto& y, size_t arg_idx)
//      {
//        size_t const ndim = x.dimension();
//        size_t el_idx = std::vector<size_t>(ndim);
//        for (int j=ndim-1; j>=0; --j) {
//          // TODO: This isn't legal, but is conceptually correct.
//          el_idx.at(j) = idx.at(
//              std::find(op_axes[arg_idx].begin(), op_axes[arg_idx].end(), j) -
//                op_axes[arg_idx].begin()
//              );
//        }
//
//        size_t const flat_idx = flatten(el_idx);
//        y = &*(x.begin() + flat_idx);
//      }
}

template<typename U, typename... Us>
void set_op_sh(U firstOp, Us... otherOps) {
  size_t const op_idx = num_ops - sizeof...(otherOps) - 1;
  for (int i=0; i<firstOp.dimension(); ++i) {
    op_sh.at(op_idx, i) = firstOp.shape().at(i);
  }
}

void set_op_sh() {
}

size_t multi_iterator::get_max_op_ndim() {
  return 0U;
}

template<typename U, typename... Us>
size_t multi_iterator::get_max_op_ndim(U firstOp, Us... otherOps) {
  return std::max(firstOp.dimension(), get_max_op_ndim(otherOps...));
}


template<typename U, typename... Us>
void multi_iterator::get_max_op_ndim(U firstOp, Us... otherOps) {
  return std::max(firstOp.dimension(), get_max_op_ndim(otherOps));
}

