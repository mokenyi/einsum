#include <algorithm>
#include <iostream>

template<size_t N = 0LU, typename... Ts>
typename std::enable_if<N == sizeof...(Ts)-1, void>::type print(
  std::tuple<Ts...> const& current,
  std::ostream& out,
  std::vector<size_t> const& idx
) {
  out << "(" << N << ":" << std::get<N>(current) << ":" << *std::get<N>(current) << ")" << std::endl;
}

template<size_t N = 0LU, typename... Ts>
typename std::enable_if<N < sizeof...(Ts)-1, void>::type print(
  std::tuple<Ts...> const& current,
  std::ostream& out,
  std::vector<size_t> const& idx
) {
  out << "(" << N << ":" << std::get<N>(current) << ":" << *std::get<N>(current) << "), ";
  print<N+1>(current, out, idx);
}

template<typename... Ts>
void multi_iterator<Ts...>::print_current() {
  std::cout << "#";
  for (int i=0; i<ndim; ++i) {
    std::cout << " " << idx.at(i);
  }
  std::cout << std::endl;
  print(current, std::cout, idx);
}

template<size_t N = 0LU, typename... Ts>
typename std::enable_if<N == sizeof...(Ts)-1, void>::type init_current(
  std::tuple<typename Ts::value_type*...>& current,
  std::tuple<Ts...>& exp
) {
  std::get<N>(current) = std::get<N>(exp).data();
}

template<size_t N = 0LU, typename... Ts>
typename std::enable_if<N < sizeof...(Ts)-1, void>::type init_current(
  std::tuple<typename Ts::value_type*...>& current,
  std::tuple<Ts...>& exp
) {
  std::get<N>(current) = std::get<N>(exp).data();
  init_current<N+1>(current, exp);
}

template<size_t N = 0LU, typename... Ts>
typename std::enable_if<N == sizeof...(Ts)-1, size_t>::type get_max_op_ndim(
  std::tuple<Ts...> const& x
) {
  return std::get<N>(x).dimension();
}

template<size_t N = 0LU, typename... Ts>
typename std::enable_if<N < sizeof...(Ts)-1, size_t>::type get_max_op_ndim(
  std::tuple<Ts...> const& x
) {
  return std::max(std::get<N>(x).dimension(), get_max_op_ndim<N+1>(x));
}

template<size_t N = 0LU, typename... Ts>
typename std::enable_if<N == sizeof...(Ts)-1, void>::type set_op_sh(
  xt::xtensor<int,2>& op_sh,
  std::tuple<Ts...> const& x
) {
  auto op = std::get<N>(x);
  for (int i=0; i<op.dimension(); ++i) {
    op_sh.at(N, i) = op.shape()[i];
  }
}

template<size_t N = 0LU, typename... Ts>
typename std::enable_if<N < sizeof...(Ts)-1, void>::type set_op_sh(
  xt::xtensor<int,2>& op_sh,
  std::tuple<Ts...> const& x
) {
  auto op = std::get<N>(x);
  for (int i=0; i<op.dimension(); ++i) {
    op_sh.at(N, i) = op.shape()[i];
  }
  set_op_sh<N+1>(op_sh, x);
}

template<typename... Ts>
multi_iterator<Ts...>::multi_iterator(
  std::tuple<Ts...> const& x,
  xt::xtensor<int,2> const& op_axes
)
: exp(x)
, op_axes(op_axes)
, ndim(op_axes.shape(1))
, idx(ndim, 0LU)
{
  op_idx.fill(0LU);
  std::array<size_t,2> op_sh_sh;
  op_sh_sh.at(0) = num_ops;
  op_sh_sh.at(1) = get_max_op_ndim(x);
  op_sh.resize(op_sh_sh);
  op_sh.fill(-1);

  set_op_sh(op_sh, x);
  set_shape();
  init_current(current, exp);
}

template<typename... Ts>
void multi_iterator<Ts...>::set_shape() {
  std::array<size_t,num_ops> i;
  std::iota(i.begin(), i.end(), 0);
  shape.resize(ndim);
  for (size_t j=0; j<ndim; ++j) {
    // Points to first operand which has a non-broadcast axis that maps
    // to dim j of the iterator.
    auto non_broadcast_op_it = std::find_if(
      i.begin(),
      i.end(),
      [this, j](size_t i_p) {
        return op_axes.at(i_p).at(j) != -1;
      }
    );
    size_t const size_dim_j = op_sh.at(
      *non_broadcast_op_it,
      op_axes.at(*non_broadcast_op_it).at(j)
    );
    auto bad_op_it = std::find_if_not(i.begin(), i.end(),
      [this, j, size_dim_j](size_t i_p) {
        return op_axes.at(i_p).at(j) == -1 || op_sh.at(i_p, op_axes.at(i_p).at(j)) == size_dim_j;
      });

    if (bad_op_it != i.end()) {
      std::stringstream ss;
      size_t const bad_axis = op_axes.at(*bad_op_it).at(j);
      ss << "For dimension " << j << " of iterator, axis " << bad_axis;
      ss << " of operand " << *bad_op_it;
      ss << " has an incorrect length " << op_sh.at(*bad_op_it, bad_axis);
      ss << " != " << size_dim_j;
      throw std::invalid_argument(ss.str());
    }

    shape.at(j) = size_dim_j;
  }
}

template<size_t I, typename... Ts>
typename std::enable_if<I == 0, void>::type set_index(
  std::tuple<typename Ts::value_type*...>& current,
  std::tuple<Ts...>& exp,
  xt::xtensor<int,2> const& op_axes,
  std::vector<size_t> const& idx)
{
  auto& op = std::get<I>(exp);
  size_t const ndim=op_axes.shape(1);
  size_t offset = 0LU;
  for (int j=0; j<ndim; ++j) {
    if (op_axes.at(I).at(j) != -1) {
      offset += op.strides()[op_axes.at(I).at(j)] * idx.at(j);
    }
  }
  std::get<I>(current) = op.data() + offset;
}

template<size_t I, typename... Ts>
typename std::enable_if<(I > 0), void>::type set_index(
  std::tuple<typename Ts::value_type*...>& current,
  std::tuple<Ts...>& exp,
  xt::xtensor<int,2> const& op_axes,
  std::vector<size_t> const& idx)
{
  auto& op = std::get<I>(exp);
  size_t const ndim=op_axes.shape(1);
  size_t offset = 0LU;
  for (int j=0; j<ndim; ++j) {
    if (op_axes.at(I).at(j) != -1) {
      offset += op.strides()[op_axes.at(I).at(j)] * idx.at(j);
    }
  }
  std::get<I>(current) = op.data() + offset;
  set_index<I-1>(current, exp, op_axes, idx);
}

template<typename... Ts>
bool multi_iterator<Ts...>::hasnext() {
  std::vector<size_t> i(ndim);
  std::iota(i.begin(), i.end(), 0LU);
  return std::any_of(i.begin(), i.end(), [this](size_t i_p) {
    return idx.at(i_p) < shape.at(i_p) - 1;
  });
}

template<typename... Ts>
void multi_iterator<Ts...>::next()
{
  if (!hasnext()) {
    throw std::out_of_range("multi_iterator out of range!");
  }

  for (int j=static_cast<int>(ndim-1); j>=0; --j)
  {
    if (idx.at(j) < shape.at(j)-1)
    {
      ++idx.at(j);
      break;
    }
    else
    {

      idx.at(j) = 0;
    }
  }
  set_index<num_ops-1>(current, exp, op_axes, idx);
}
