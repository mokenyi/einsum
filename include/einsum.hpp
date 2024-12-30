#ifndef EINSUM_HPP
#define EINSUM_HPP
#include <tuple>
#include <string>
#include "xtensor/xexpression.hpp"

#define MAXDIMS 10

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

template<typename T>
T get_combined_dims_view(T const& op, int iop, std::array<char,MAXDIMS> const& labels) {
  
}

template<size_t I = 0LU, typename... Ts>
typename std::enable_if<I == sizeof...(Ts),void>::type get_combined_dims_view_loop(
  std::tuple<Ts...>& ops,
  std::array<std::array<char,MAXDIMS>,sizeof...(Ts)> op_labels
) {
}

template<size_t I = 0LU, typename... Ts>
typename std::enable_if<I < sizeof...(Ts),void>::type get_combined_dims_view_loop(
  std::tuple<Ts...>& ops,
  std::array<std::array<char,MAXDIMS>,sizeof...(Ts)> op_labels
) {
  std::get<I>(ops) = get_combined_dims_view(std::get<I>(ops), I, op_labels.at(I));
  get_combined_dims_view_loop<I+1>(ops, op_labels);
}

template<typename... Ts>
auto einsum(std::string const& subscripts, xt::xexpression<Ts> const&... op_in) {
  auto ops = std::make_tuple(op_in.derived_cast()...);
  std::array<char,128> label_counts;
  label_counts.fill(0);

  size_t const nop = sizeof...(op_in);
  auto subscripts_begin = subscripts.begin();

  std::array<std::array<char,MAXDIMS>,nop> op_labels;
  int min_label(INT_MAX);
  int max_label(INT_MIN);

  parse_operand_subscripts_loop(
    subscripts_begin,
    subscripts.end(),
    ops,
    op_labels,
    label_counts,
    min_label,
    max_label
  );
  
  int const ndim_broadcast = get_ndim_broadcast_loop(ops, op_labels);
  std::array<char,MAXDIMS> output_labels;
  int ndim_output(-1);
  if (*subscripts_begin == '\0') {
    std::fill(
      output_labels.begin(),
      std::next(output_labels.begin(),ndim_broadcast),
      0
    );
    ndim_output = ndim_broadcast;
    for (int label = min_label; label <= max_label; ++label) {
      if (label_counts.at(label) == 1) {
        if (ndim_output < MAXDIMS) {
          output_labels.at(ndim_output++) = label;
        }
        else {
          throw std::invalid_argument("Einstein sum subscript string has too "
            "many distinct layers");
        }
      }
    }
  }
  else {
    if (*subscripts_begin != '-' || *std::next(subscripts_begin) != '>') {
      throw std::invalid_argument("Einstein sub subscript string does not "
        "contain proper '->' output
    }
    std::advance(subscripts_begin, 2);
    ndim_output = parse_output_subscripts(
      subscripts_begin,
      subscripts.end(),
      ndim_broadcast,
      label_counts,
      output_labels
    );
  }

  get_combined_dims_view_loop(ops, op_labels);
}

#endif // EINSUM_HPP
