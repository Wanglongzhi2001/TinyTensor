

class dt_byte {
    unsigned char _;

public:
    //! convert to given type
    template <typename T>
    T* as() {
        return reinterpret_cast<T*>(this);
    }

    //! convert to given type
    template <typename T>
    const T* as() const {
        return reinterpret_cast<const T*>(this);
    }
};
