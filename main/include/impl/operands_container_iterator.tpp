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

template<size_t I, typename... Ts>
typename std::enable_if<I == 0, void>::type set_index(
  std::tuple<typename Ts::value_type*...>& current,
  std::tuple<Ts...>& exp,
  typename operands_container<Ts...>::op_axes_type const& op_axes,
  std::vector<size_t> const& idx)
{
  auto& op = std::get<I>(exp);
  size_t const ndim=op_axes.at(0).size();
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
  typename operands_container<Ts...>::op_axes_type const& op_axes,
  std::vector<size_t> const& idx)
{
  auto& op = std::get<I>(exp);
  size_t const ndim=op_axes.at(0).size();
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
operands_container<Ts...>::iterator::iterator (
  operands_container<Ts...>& parent,
  std::vector<size_t>&& idx,
  bool is_end
)
: parent(parent)
, idx(idx)
, is_end(is_end)
{
  set_index<num_ops-1>(current, parent.exp, parent.op_axes, idx);
}


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
void operands_container<Ts...>::iterator::print_current() {
  std::cout << "#";
  for (int i=0; i<parent.ndim; ++i) {
    std::cout << " " << idx.at(i);
  }
  std::cout << std::endl;
  print(current, std::cout, idx);
}

template<typename... Ts>
typename operands_container<Ts...>::iterator& operands_container<Ts...>::iterator::operator++() {
  for (int j=static_cast<int>(parent.ndim-1); j>=0; --j)
  {
    if (idx.at(j) < parent.shape.at(j)-1)
    {
      ++idx.at(j);
      break;
    }
    else
    {
      idx.at(j) = 0;
      if (j == 0) {
        for (int i=0; i<parent.ndim; ++i) { std::cout << " " << idx.at(i); }
        std::cout << std::endl;
        is_end = true;
      }
    }
  }
  set_index<num_ops-1>(current, parent.exp, parent.op_axes, idx);
  return *this;
}

template<typename... Ts>
bool operands_container<Ts...>::iterator::operator==(operands_container<Ts...>::iterator const& other) {
  return current == other.current && is_end == other.is_end;
}

template<typename... Ts>
bool operands_container<Ts...>::iterator::operator!=(operands_container<Ts...>::iterator const& other) {
  return !(*this == other);
}
