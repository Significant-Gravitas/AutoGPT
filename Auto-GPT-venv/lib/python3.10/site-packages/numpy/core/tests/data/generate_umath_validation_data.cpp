#include <algorithm>
#include <fstream>
#include <iostream>
#include <cmath>
#include <random>
#include <cstdio>
#include <ctime>
#include <vector>

struct ufunc {
    std::string name;
    double (*f32func)(double);
    long double (*f64func)(long double);
    float f32ulp;
    float f64ulp;
};

template <typename T>
T
RandomFloat(T a, T b)
{
    T random = ((T)rand()) / (T)RAND_MAX;
    T diff = b - a;
    T r = random * diff;
    return a + r;
}

template <typename T>
void
append_random_array(std::vector<T> &arr, T min, T max, size_t N)
{
    for (size_t ii = 0; ii < N; ++ii)
        arr.emplace_back(RandomFloat<T>(min, max));
}

template <typename T1, typename T2>
std::vector<T1>
computeTrueVal(const std::vector<T1> &in, T2 (*mathfunc)(T2))
{
    std::vector<T1> out;
    for (T1 elem : in) {
        T2 elem_d = (T2)elem;
        T1 out_elem = (T1)mathfunc(elem_d);
        out.emplace_back(out_elem);
    }
    return out;
}

/*
 * FP range:
 * [-inf, -maxflt, -1., -minflt, -minden, 0., minden, minflt, 1., maxflt, inf]
 */

#define MINDEN std::numeric_limits<T>::denorm_min()
#define MINFLT std::numeric_limits<T>::min()
#define MAXFLT std::numeric_limits<T>::max()
#define INF std::numeric_limits<T>::infinity()
#define qNAN std::numeric_limits<T>::quiet_NaN()
#define sNAN std::numeric_limits<T>::signaling_NaN()

template <typename T>
std::vector<T>
generate_input_vector(std::string func)
{
    std::vector<T> input = {MINDEN,  -MINDEN, MINFLT, -MINFLT, MAXFLT,
                            -MAXFLT, INF,     -INF,   qNAN,    sNAN,
                            -1.0,    1.0,     0.0,    -0.0};

    // [-1.0, 1.0]
    if ((func == "arcsin") || (func == "arccos") || (func == "arctanh")) {
        append_random_array<T>(input, -1.0, 1.0, 700);
    }
    // (0.0, INF]
    else if ((func == "log2") || (func == "log10")) {
        append_random_array<T>(input, 0.0, 1.0, 200);
        append_random_array<T>(input, MINDEN, MINFLT, 200);
        append_random_array<T>(input, MINFLT, 1.0, 200);
        append_random_array<T>(input, 1.0, MAXFLT, 200);
    }
    // (-1.0, INF]
    else if (func == "log1p") {
        append_random_array<T>(input, -1.0, 1.0, 200);
        append_random_array<T>(input, -MINFLT, -MINDEN, 100);
        append_random_array<T>(input, -1.0, -MINFLT, 100);
        append_random_array<T>(input, MINDEN, MINFLT, 100);
        append_random_array<T>(input, MINFLT, 1.0, 100);
        append_random_array<T>(input, 1.0, MAXFLT, 100);
    }
    // [1.0, INF]
    else if (func == "arccosh") {
        append_random_array<T>(input, 1.0, 2.0, 400);
        append_random_array<T>(input, 2.0, MAXFLT, 300);
    }
    // [-INF, INF]
    else {
        append_random_array<T>(input, -1.0, 1.0, 100);
        append_random_array<T>(input, MINDEN, MINFLT, 100);
        append_random_array<T>(input, -MINFLT, -MINDEN, 100);
        append_random_array<T>(input, MINFLT, 1.0, 100);
        append_random_array<T>(input, -1.0, -MINFLT, 100);
        append_random_array<T>(input, 1.0, MAXFLT, 100);
        append_random_array<T>(input, -MAXFLT, -100.0, 100);
    }

    std::random_shuffle(input.begin(), input.end());
    return input;
}

int
main()
{
    srand(42);
    std::vector<struct ufunc> umathfunc = {
            {"sin", sin, sin, 2.37, 3.3},
            {"cos", cos, cos, 2.36, 3.38},
            {"tan", tan, tan, 3.91, 3.93},
            {"arcsin", asin, asin, 3.12, 2.55},
            {"arccos", acos, acos, 2.1, 1.67},
            {"arctan", atan, atan, 2.3, 2.52},
            {"sinh", sinh, sinh, 1.55, 1.89},
            {"cosh", cosh, cosh, 2.48, 1.97},
            {"tanh", tanh, tanh, 1.38, 1.19},
            {"arcsinh", asinh, asinh, 1.01, 1.48},
            {"arccosh", acosh, acosh, 1.16, 1.05},
            {"arctanh", atanh, atanh, 1.45, 1.46},
            {"cbrt", cbrt, cbrt, 1.94, 1.82},
            //{"exp",exp,exp,3.76,1.53},
            {"exp2", exp2, exp2, 1.01, 1.04},
            {"expm1", expm1, expm1, 2.62, 2.1},
            //{"log",log,log,1.84,1.67},
            {"log10", log10, log10, 3.5, 1.92},
            {"log1p", log1p, log1p, 1.96, 1.93},
            {"log2", log2, log2, 2.12, 1.84},
    };

    for (int ii = 0; ii < umathfunc.size(); ++ii) {
        // ignore sin/cos
        if ((umathfunc[ii].name != "sin") && (umathfunc[ii].name != "cos")) {
            std::string fileName =
                    "umath-validation-set-" + umathfunc[ii].name + ".csv";
            std::ofstream txtOut;
            txtOut.open(fileName, std::ofstream::trunc);
            txtOut << "dtype,input,output,ulperrortol" << std::endl;

            // Single Precision
            auto f32in = generate_input_vector<float>(umathfunc[ii].name);
            auto f32out = computeTrueVal<float, double>(f32in,
                                                        umathfunc[ii].f32func);
            for (int jj = 0; jj < f32in.size(); ++jj) {
                txtOut << "np.float32" << std::hex << ",0x"
                       << *reinterpret_cast<uint32_t *>(&f32in[jj]) << ",0x"
                       << *reinterpret_cast<uint32_t *>(&f32out[jj]) << ","
                       << ceil(umathfunc[ii].f32ulp) << std::endl;
            }

            // Double Precision
            auto f64in = generate_input_vector<double>(umathfunc[ii].name);
            auto f64out = computeTrueVal<double, long double>(
                    f64in, umathfunc[ii].f64func);
            for (int jj = 0; jj < f64in.size(); ++jj) {
                txtOut << "np.float64" << std::hex << ",0x"
                       << *reinterpret_cast<uint64_t *>(&f64in[jj]) << ",0x"
                       << *reinterpret_cast<uint64_t *>(&f64out[jj]) << ","
                       << ceil(umathfunc[ii].f64ulp) << std::endl;
            }
            txtOut.close();
        }
    }
    return 0;
}
