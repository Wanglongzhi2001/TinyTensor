#pragma once

#define TT_CHECK(...) \
    for (bool _check_status = (__VA_ARGS__); !_check_status;) \
        throw std::runtime_error("Error occured at: " #__VA_ARGS__ " at " __FILE__ ":" + std::to_string(__LINE__));

#define TT_UNIMPLEMENTED() throw std::runtime_error("Unimplemented at " + std::to_string(__LINE__) + " in " __FILE__);

#define TT_ASSERT(cond, msg)    \
    if (!(cond)) {              \
        throw std::runtime_error(msg); \
    }

#define TT_THROW(_exc, _msg...) throw _exc(_msg);

//! throw exception with given message if condition is true
#define TT_THROW_IF(_cond, _exc, _msg...) \
    do {                                   \
        if (static_cast<bool>(_cond))         \
            tt_throw(_exc, _msg);         \
    } while (0)

//! branch prediction hint: unlikely to take
#define tt_unlikely(v) __builtin_expect(static_cast<bool>(v), 0)