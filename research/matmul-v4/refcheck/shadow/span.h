// Minimal shadow of BTX span.h — 2-arg + array ctor, data()/size().
#ifndef BTX_SHADOW_SPAN_H
#define BTX_SHADOW_SPAN_H
#include <cstddef>
template <class T>
class Span {
    T* m_data;
    std::size_t m_size;
public:
    constexpr Span() : m_data(nullptr), m_size(0) {}
    constexpr Span(T* d, std::size_t s) : m_data(d), m_size(s) {}
    template <std::size_t N> constexpr Span(T (&a)[N]) : m_data(a), m_size(N) {}
    constexpr T* data() const { return m_data; }
    constexpr std::size_t size() const { return m_size; }
    constexpr T* begin() const { return m_data; }
    constexpr T* end() const { return m_data + m_size; }
};
#endif
