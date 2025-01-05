#ifndef EINSUM_HPP
#define EINSUM_HPP
#include <tuple>
#include <string>
#include "xtensor/xexpression.hpp"

#define MAXDIMS 10

template<typename O, typename... Ss>
einsum<Ss>::einsum(int e)
: ellipsis(e)
{
}

template<typename T>
void parse_operand_subscripts_impl(
  std::string::iterator& subscripts_begin,
  std::string::iterator subscripts_end,
  size_t ndim,
  size_t iop,
  std::array<char,MAXDIMS>& op_labels,
  std::array<char,128>& label_counts,
  int& min_label,
  int& max_label)
{
  int idim(0);
  int ellipsis(-1);
  size_t const length = std::distance(subscripts_begin, subscripts_end);

  for (auto it = subscripts_begin; it != subscripts_end; ++it) {
    size_t const i = std::distance(subscripts_begin, it);
    int const label = *it;
    if (label > 0 && isalpha(label)) {
      if (idim >= ndim) {
        throw std::invalid_argument("Einstein sum subscripts string "
          "contains too many subscripts for operand " + std::tostring(iop));
      }

      op_labels.at(idim++) = label;
      if (label < min_label) {
        min_label = label;
      }
      if (label > max_label) {
        max_label = label;
      }
      label_counts.at(label)++;
    }
    else if (label == '.') {
      if (ellipsis != -1 || i + 2 >= length || *(++it) != '.' || *(++it) != '.') {
        throw std::invalid_argument("Einstein sum subscripts string "
          "contains a '.' that is not part of an ellipsis ('...') "
          "in operand " + std::tostring(iop));
      }

      ellipsis = idim;
    }
    else if (label != ' ') {
      std::stringstream ss;
      ss << "Invalid subscript " << static_cast<char>(label) << " in Einstein ";
      ss << "subscripts string, subscripts must be letters"
      throw std::invalid_argument(ss.str());
    }
  }

  if (ellipsis == -1) {
    throw std::invalid_argument("Operand has more dimensions than subscripts "
      "given in einstein sum, but no '...' ellipsis provided to broadcast the "
      "extra dimensions.");
  }
  else if (idim < ndim) {
    for (int i = 0; i < idim - ellipsis; ++i) {
        op_labels.at(ndim - i - 1) = op_labels.at(idim - i - 1);
    }
    /* Set all broadcast dimensions to zero. */
    for (int i = 0; i < ndim - idim; ++i) {
        op_labels.at(ellipsis + i) = 0;
    }
  }

  for (int idim = 0; idim < ndim - 1; ++idim) {
    int const label = (signed char)op_labels.at(idim);
    /* If it is a proper label, find any duplicates of it. */
    if (label > 0) {
      /* Search for the next matching label. */
      char *next = memchr(op_labels + idim + 1, label, ndim - idim - 1);

      while (next != NULL) {
        /* The offset from next to op_labels[idim] (negative). */
        *next = (char)((op_labels + idim) - next);
        /* Search for the next matching label. */
        next = memchr(next + 1, label, op_labels + ndim - 1 - next);
      }
    }
  }

  return 0;
}


template<typename T>
void parse_operand_subscripts(
  std::string::iterator& subscripts_begin,
  std::string::iterator subscripts_end,
  size_t iop,
  size_t nop,
  T const& op,
  std::array<char,MAXDIMS>& op_labels,
  std::array<char,128>& label_counts,
  int& min_label,
  int& max_label) {

  auto const op_subscripts_end = std::find_if_not(subscripts_begin, subscripts_end,
    [](char c) {
      return c == ',' || c == '-';
    });
  size_t const length = std::distance(subscripts_begin, op_subscripts_end);
  if (iop == nop-1 && *subscripts_end == ',') {
    throw std::invalid_argument("More operands provided to Einstein sum "
      "function than specified in the subscripts string");
  }

  if (iop < nop-1 && *subscripts_end != ',') {
    throw std::invalid_argument("Fewer operands provided to Einstein sum "
      "function than specified in the subscripts string");
  }

  parse_operand_subscripts_impl(subscripts_begin, subscripts_end, op.dimension(), iop, op_labels.at(iop), label_counts, min_label, max_label);

  subscripts_begin = op_subscripts_end;

  if (iop < nop-1) {
    ++subscripts_begin;
  }
}

template<size_t I, typename... Ts>
typename std::enable_if<I == sizeof...(Ts), void>::type parse_operand_subscripts_loop(
  std::string::iterator begin,
  std::string::iterator end,
  std::tuple<Ts...> const& ops,
  std::array<std::array<char,MAXDIMS>,sizeof...(Ts)>& op_labels,
  std::array<char,128>& label_counts,
  int& min_label,
  int& max_label) {
}

template<size_t I = 0LU, typename... Ts>
typename std::enable_if<I < sizeof...(Ts), void>::type parse_operand_subscripts_loop(
  std::string::iterator& begin,
  std::string::iterator end,
  std::tuple<Ts...> const& ops,
  std::array<std::array<char,MAXDIMS>,sizeof...(Ts)>& op_labels,
  std::array<char,128>& label_counts,
  int& min_label,
  int& max_label) {

  parse_operand_subscripts(
    begin,
    end,
    I,
    sizeof...(Ts),
    std::get<I>(ops),
    op_labels.at(I),
    label_counts,
    min_label,
    max_label);

  parse_operand_subscripts_loop<I+1>(
    begin,
    end,
    ops,
    op_labels,
    label_counts,
    min_label,
    max_label
  );
}

template<typename T>
size_t get_ndim_broadcast(
  T const& op,
  std::array<char,MAXDIMS> const& labels
) {
  int const ndim = op.dimension();
  return std::count(labels.begin(), std::next(labels.begin(),ndim), 0);
}


template<size_t I = 0LU, typename... Ts>
typename std::enable_if<size_t I == sizeof...(Ts), size_t>::type get_ndim_broadcast_loop(
  std::tuple<Ts...> const& ops,
  std::array<std::array<char,MAXDIMS>,nop> const& op_labels
) {
}

template<size_t I = 0LU, typename... Ts>
typename std::enable_if<size_t I < sizeof...(Ts), size_t>::type get_ndim_broadcast_loop(
  std::tuple<Ts...> const& ops,
  std::array<std::array<char,MAXDIMS>,sizeof...(Ts)> const& op_labels
) {
  return std::max(
    get_ndim_broadcast(std::get<I>(ops), op_labels.at(I)),
    get_ndim_broadcast_loop<I+1>(ops, op_labels)
  );
}

int parse_output_subscripts(
  std::string::iterator subscripts_begin,
  std::string::iterator subscripts_end,
  int ndim_broadcast,
  std::array<char,128>& label_counts,
  std::array<char,MAXDIMS>& output_labels
) {
  int ndim(0);
  bool ellipsis(false);
  for (auto it = subscripts_begin; it != subscripts_end; ++it) {
    int const label = *it;
    if (label > 0 && std::isalpha(label)) {
      if (std::find(std::next(it),subscripts_end,label) != subscripts_end) {
        std::stringstream ss;
        ss << "Einstein sum subscripts string includes output subscript '";
        ss << static_cast<char>(label) << "' multiple times";
        throw std::invalid_argument(ss.str());
      }

      if (label_counts.at(label) == 0) {
        std::stringstream ss;
        ss << "Einstein sum subscripts string included output subscript '";
        ss << static_cast<char>(label) << "' which never appeared in an input";
        throw std::invalid_argument(ss.str());
      }

      if (ndim >= MAXDIMS) {
        throw std::invalid_argument("Einstein sum subscripts string "
          "contains too many subscripts in the output");
      }

      output_labels.at(ndim++) = label;
    }
    else if (label == '.') {

      if (ellipsis || std::distance(it,subscripts_end) <= 2 ||
        std::any_of(it, std::next(it,3), [](char c) { return c != '.'; })) {
        throw std::invalid_argument("Einstein sum subscripts string contains "
          "a '.' that is not part of an ellipsis ('...') in the output");
      }

      if (ndim + ndim_broadcast > MAXDIMS) {
        throw std::invalid_argument("Einstein sub subscripts string contains "
          "too many subscripts in the output");
      }

      ellipsis = true;
      std::advance(it,2);
      for (int bdim = 0; bdim < ndim_broadcast; ++bdim) {
        output_labels.at(ndim++) = 0;
      }
    }
    else if (label != ' ') {
      std::stringstream ss;
      ss << "Invalid subscript '" << static_cast<char>(label) << "' in ";
      ss << "Einstein sum subscripts string, subscripts must be letters";

      throw std::invalid_argument(ss.str());
    }
  }

  if (!ellipsis && ndim_broadcast > 0) {
    throw std::invalid_argument("Output has more dimensions than subscripts "
      "given in Einstein sum, but no '...' ellipsis provided to broadcast the "
      "extra dimensions");
  }

  return ndim;
}

template<int I = 0, typename... Ss>
typename std::enable_if<I == sizeof...(Ss), void>::type get_label_counts_loop(
  std::tuple<Ss...> const& subs,
  std::vector<std::array<int,2>>& label_counts,
  int ellipsis
) {
}

template<int I = 0, typename... Ss>
typename std::enable_if<I < sizeof...(Ss), void>::type get_label_counts_loop(
  std::tuple<Ss...> const& subs,
  std::vector<std::array<int,2>>& label_counts,
  int ellipsis
) {
  std::vector<int> const labels = (typename std::tuple_element<I, std::tuple<Ss...>>::type)::get_combined_labels();

  for (int k=0; k<labels.size(); ++k) {
    int const i = labels.at(k);

    auto it = std::find_if(
      label_counts.begin(),
      label_counts.end(),
      [i](std::array<int,2> const& a) { return a.at(0) == i; }
    );

    if (it != label_counts.end()) {
      it->at(1)++;
    }

    else {
      label_counts.push_back(std::array<int,2>());
      label_counts.back().at(0) = i;
      label_counts.back().at(1) = 1;
    }
  }
  
  get_label_counts_loop<I+1>(subs, label_counts, ellipsis);
}

template<typename O>
typename std::enable_if<
  !std::is_same<O,implicit_out>::value,
  std::vector<int>,
  int ellipsis
>::type get_output_labels(
  std::vector<std::array<int,2>> const& label_counts
  int ndim_broadcast,
  int ellipsis
) {
  static_assert(
    !has_repeated_labels<O::start_first,O::start_second,T>::value
    "output subscripts must not have repeated labels"
  );

  std::vector<int> labels = O::to_vector();
  int const length = labels.size();

  int const num_ellipses = std::count(labels.begin(), labels.end(), ellipsis);
  if (num_ellipses > 1) {
    std::stringstream ss;
    ss << "Found multiple ellipses in subscripts " << to_string();
    throw std::invalid_argument(ss.str());
  }

  auto ellipsis_it = std::find(labels.begin(), labels.end(), ellipsis);
  int const found_ellipsis = ellipsis_it == labels.end() ? -1 :
    std::distance(labels.begin(), ellipsis_it);

  if (found_ellipsis != -1) {
    for (int i=0; i<ndim_broadcast-1; ++i) {
      labels.insert(std::next(labels.begin(),found_ellipsis),ellipsis);
    }
  }
  else if (ndim_broadcast > 0) {
    throw std::invalid_argument("Output has more dimensions than subscripts "
      "given in Einstein sum, but no ellipsis provided to broadcast the "
      "extra dimensions");
  }

  for (int i=0; i<length; ++i) {
    int const label = output_labels.at(i);
    if (label != ellipsis) {
      auto label_count_it = std::find_if(
        label_counts.begin(),
        label_counts.end(),
        [label](std::array<int,2> const& l) { return l.at(0) == label; }
      );
      if (label_count_it == label_counts.end()) {
        std::stringstream ss;
        ss << "Output subscripts contain label " << label << " which does not ";
        ss << "appear in input subscripts";
        throw std::invalid_argument(ss.str());
      }
    }
  }

  return output_labels;
}

template<typename O>
typename std::enable_if<
  std::is_same<O,implicit_out>::value,
  std::vector<int>,
>::type get_output_labels(
  std::vector<std::array<int,2>> const& label_counts,
  int ndim_broadcast,
  int ellipsis
) {
  std::vector<int> output_labels(ndim_broadcast, ellipsis);
  for (auto const& label_count: label_counts) {
    if (label_count.at(1) == 1) {
      output_labels.push_back(label_count.at(0));
    }
  }
  return output_labels;
}

template<size_t I = 0lu, typename U, typename... Ts>
typename std::enable_if<I == sizeof...(Ts),void>::type copy(
  std::vector<U>& dest,
  std::tuple<Ts...> const& src
) {
}

template<size_t I = 0lu, typename U, typename... Ts>
typename std::enable_if<I < sizeof...(Ts),void>::type copy(
  std::array<U,sizeof...(Ts)>& dest,
  std::tuple<Ts...> const& src
) {
  dest.at(I) = std::get<I>(src);
  as_vector<I+1>(dest, in);
}

template<size_t I=0lu, typename... Ts>
typename std::enable_if<I == sizeof...(Ts),void>::type prepare_op_axes_loop(
  std::tuple<Ts...> const& ops,
  std::array<std::vector<int>,sizeof...(Ts)> const& combined_labels,
  std::array<std::vector<int>,sizeof...(Ts)+1>& op_axes,
  std::vector<int> const& iter_labels,
  int ellipsis
) {
}

template<size_t I=0lu, typename... Ts>
typename std::enable_if<I < sizeof...(Ts),void>::type prepare_op_axes_loop(
  std::tuple<Ts...> const& ops,
  std::array<std::vector<int>,sizeof...(Ts)> const& combined_labels,
  std::array<std::vector<int>,sizeof...(Ts)+1>& op_axes,
  std::vector<int> const& iter_labels,
  int ellipsis
) {

  int const ndim_op = std::get<I>(ops).dimension();
  std::vector<int> const& axes = op_axes.at(I);
  std::vector<int> const& labels = combined_labels.at(I);
  int const ndim_iter = iter_labels.size();

  axes.resize(ndim_iter, -1);

  int ibroadcast = ndim_op-1;
  for (int i=ndim_iter-1; i>=0; --i) {
    int const label = iter_labels.at(i);
    if (label == ellipsis) {
      while (ibroadcast >= 0 && labels.at(ibroadcast) != 0) {
        --ibroadcast;
      }

      if (ibroadcast < 0) {
        axes.at(i) = -1;
      }
      else {
        axes.at(i) = ibroadcast;
        --ibroadcast;
      }
    }
    else {
      auto const match = std::find(labels.begin(), labels.end(), label);
      if (match == labels.end()) {
        axes.at(i) = -1;
      }
      else {
        axes.at(i) = std::distance(labels.begin(), match);
      }
    }
  }

  prepare_op_axes_loop<I+1>(
    ops,
    combined_labels,
    op_axes,
    iter_labels,
    ellipsis
  );
}

template<typename O, typename... Ss>
template<typename... Ts>
auto einsum<Ss>::eval(xt::xexpression<Ts> const&... op_in) -> xt::xarray<typename std::common_type<Ts::value_type...>::type> {
  // TODO: Consider view_eval(op_in.derived_cast()) to allow op_in to work
  // with general xexpressions.

  auto ops = std::make_tuple(
    get_combined_dims_view<Ss...>(
      op_in,
      ellipsis
    ).derived_cast()...
  );

  std::array<char,128> label_counts;
  label_counts.fill(0);

  std::vector<std::array<int,2>> label_counts;

  auto const subs = std::make_tuple(Ss()...);
  get_label_counts_loop(subs, label_counts, ellipsis);

  constexpr size_t num_ops = sizeof...(ops);
  
  std::array<std::vector<int>,num_ops> combined_labels;

  copy(
    combined_labels,
    std::make_tuple(Ss::get_combined_labels(op_in.dimension(),ellipsis)...)
  );

  int ndim_broadcast(-1);
  for (auto const& combined_label: combined_labels) {
    int const num_zeros = std::count(
      combined_label.begin(),
      combined_label.end(),
      ellipsis
    );

    ndim_broadcast = num_zeros > ndim_broadcast ? num_zeros : ndim_broadcast;
  }

  std::vector<int> const output_labels = get_output_labels<O>(
    label_counts,
    ndim_broadcast,
    ellipsis
  );

  std::vector<int> iter_labels(output_labels);
  for (auto const& label_count: label_counts){
    iter_labels.push_back(label_count.at(0));
  }

  int const ndim_iter = iter_labels.size();

  std::array<std::vector<int>,num_ops+1> op_axes;
  prepare_op_axes_loop(ops,combined_labels,op_axes,iter_labels,ellipsis);

  std::vector<int>& output_op_axes = op_axes.at(num_ops);
  output_op_axes.resize(ndim_iter);
  std::iota(
    output_op_axes.begin(),
    std::next(output_op_axes.begin(),ndim_output),
    0
  );

  std::fill(
    std::next(output_op_axes.begin(),ndim_output),
    output_op_axes.end(),
    -1
  );


  auto iter = multi_iterator(ops, op_axes);
 
  using output_value_type = typename std::common_type<Ts::value_type...>::type;
  // TODO: Do this for real!
  return xt::xarray<output_value_type>{{0.0, 0.0}}; 
}

#endif // EINSUM_HPP
