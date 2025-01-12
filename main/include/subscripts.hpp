#ifndef SUBSCRIPTS_HPP
#define SUBSCRIPTS_HPP
#include <cstdlib>
#include <tuple>
#include <vector>

template<size_t I = 0lu, typename... Ts>
typename std::enable_if<I == sizeof...(Ts), void>::type push_value(std::vector<int>& dest, std::tuple<Ts...> const& src) {
}

template<size_t I = 0lu, typename... Ts>
typename std::enable_if<I < sizeof...(Ts), void>::type push_value(std::vector<int>& dest, std::tuple<Ts...> const& src) {
  dest.push_back(std::tuple_element<I, std::tuple<Ts...>>::type::value);
  push_value<I+1>(dest, src);
}

template<int... I>
struct subscripts {
  static constexpr size_t start_first = sizeof...(I) - 2;
  static constexpr size_t start_second = sizeof...(I) - 1;
  using tuple_type = std::tuple<std::integral_constant<int,I>...>;

  static std::string to_string() {
    std::vector<int> const labels = to_vector();
    std::stringstream ss;
    ss << "[" << *labels.begin();
    for (int j=1; j<labels.size(); ++j) {
      ss << ", " << labels.at(j);
    }
    ss << "]";

    return ss.str();
  }

  static std::vector<int> parse_operand_subscripts(
    int ndim,
    int ellipsis = static_cast<int>('_')
  ) {
    constexpr size_t num_subscripts = sizeof...(I);
    if (num_subscripts > ndim) {
      std::stringstream ss;
      ss << "num_subscripts (" << num_subscripts << ") > ndim (" << ndim;
      ss << ")";
      throw std::invalid_argument(ss.str());
    }

    std::vector<int> labels = to_vector();

    int const num_ellipses = std::count(labels.begin(), labels.end(), ellipsis);
    if (num_ellipses > 1) {
      std::stringstream ss;
      ss << "Found multiple ellipses in subscripts " << to_string();
      throw std::invalid_argument(ss.str());
    }

    auto ellipsis_it = std::find(labels.begin(), labels.end(), ellipsis);
    int const found_ellipsis = ellipsis_it == labels.end() ? -1 :
      std::distance(labels.begin(), ellipsis_it);

    if (found_ellipsis == -1) {
      if (num_subscripts != ndim) {
        std::stringstream ss;
        ss << "ndim (" << ndim << ") != number of subscripts (";
        ss << num_subscripts;
        ss << ") but no ellipsis provided to broadcast the extra dimensions";
        throw std::invalid_argument(ss.str());
      }
    }
    else {
      for (int i=0; i<ndim-num_subscripts; ++i) {
        labels.insert(std::next(labels.begin(), found_ellipsis), ellipsis);
      }
    }

    // Overwrite each repeated label with the offset to the first appearance
    // of the label.
    for (auto it=labels.begin(); it!=labels.end(); ++it) {
      if (*it == ellipsis) {
        continue;
      }
      for (
          auto jt=std::find(std::next(it),labels.end(),*it);
          jt!=labels.end();
          jt=std::find(++jt,labels.end(),*it)
      ) {
        *jt = -static_cast<int>(std::distance(it,jt));
      } 
    }

    return labels;
  }

  static std::vector<int> to_vector() {
    std::vector<int> ret;
    push_value(ret, tuple_type());
    return ret;
  }

  static std::vector<int> get_combined_labels(int ndim, int ellipsis) {
    std::vector<int> labels = parse_operand_subscripts(ndim, ellipsis);
    for (auto it=labels.begin(); it!=labels.end(); ++it) {
      if (*it < 0) {
        it = labels.erase(it);
      }
    }

    return labels;
  }
};

#endif // SUBSCRIPTS_HPP
