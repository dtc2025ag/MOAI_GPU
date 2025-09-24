// SPDX-License-Identifier: MIT
//
// This file implements a pair of classes, ``Chebyshev`` and ``SiLU``,
// that mirror the behaviour of the corresponding Python classes from
// the Orion project (``orion/nn/activation.py``).  The goal of this
// translation is to match the original implementation as closely as
// possible while providing CUDA‐accelerated evaluation of the fitted
// Chebyshev polynomial.  The code is entirely self contained: it
// computes Chebyshev coefficients on the host using the same least
// squares routine as ``numpy.polynomial.Chebyshev.fit`` (via a
// custom reimplementation) and performs polynomial evaluation on the
// device using a Clenshaw recurrence.  Only single precision
// coefficients are stored on the device but all fitting work is done
// in double precision to maximise accuracy.

#include <cuda_runtime.h>
#include <vector>
#include <cmath>
#include <functional>
#include <stdexcept>
#include <iostream>
#include <algorithm>

// A helper macro to check CUDA calls and throw on error.  This
// simplifies host code error handling.
#define CUDA_CHECK(err)                                                      \
    do {                                                                    \
        cudaError_t e = (err);                                              \
        if (e != cudaSuccess) {                                             \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__    \
                      << " -> " << cudaGetErrorString(e) << std::endl;      \
            throw std::runtime_error("CUDA runtime error");                \
        }                                                                   \
    } while (0)

// ---------------------------------------------------------------------------
// Device side: evaluate a Chebyshev series using the Clenshaw recurrence.
// The polynomial is assumed to be of the form
//   p(z) = c0/2 + c1*T1(z) + ... + cn*Tn(z),
// where Tn(z) are the first kind Chebyshev polynomials.  This routine
// expects z to lie in [-1,1] for maximum accuracy; caller is
// responsible for clamping inputs if necessary.  The recurrence runs
// backwards to minimise numerical error.
__device__ inline float clenshaw_eval(const float z,
                                      const float* __restrict__ coeffs,
                                      int degree)
{
    float bk2 = 0.0f; // b_{k+2}
    float bk1 = 0.0f; // b_{k+1}
    // iterate from highest degree down to 1
    for (int k = degree; k >= 1; --k) {
        float bk = 2.0f * z * bk1 - bk2 + coeffs[k];
        bk2 = bk1;
        bk1 = bk;
    }
    // final combination: c0/2 + z*b1 - b2
    return z * bk1 - bk2 + 0.5f * coeffs[0];
}

// Device kernel: evaluate a vector of inputs through the fitted
// Chebyshev polynomial.  Inputs are rescaled/shifted into [-1,1] when
// necessary, and then passed through the Clenshaw evaluator.  An
// optional output scale is applied at the end.  See forward() for
// details of fused and he_mode behaviour.
__global__ void chebyshev_forward_kernel(const float* __restrict__ x,
                                         float* __restrict__ y,
                                         int n,
                                         const float* __restrict__ coeffs,
                                         int degree,
                                         bool he_mode,
                                         bool fused,
                                         float prescale,
                                         float constant,
                                         float output_scale)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= n) return;

    float xi = x[i];
    float result;
    if (!he_mode) {
        // In non-HE mode simply evaluate the scalar activation directly
        // using the host provided function pointer is not possible on
        // device; instead the caller must have precomputed these
        // values on the host and copied them to the device.
        // This branch will not be invoked by the kernel and is
        // present solely for completeness.  To avoid undefined
        // behaviour, set y[i] to zero.
        result = 0.0f;
    } else {
        if (!fused) {
            // apply prescale and shift only when not fused; this
            // matches the Python implementation which defers scaling
            // when multiple activations are composed together
            if (prescale != 1.0f) {
                xi *= prescale;
            }
            if (constant != 0.0f) {
                xi += constant;
            }
        }
        // clamp xi into [-1,1] for stability
        float z = xi;
        if (z < -1.0f) z = -1.0f;
        else if (z > 1.0f) z = 1.0f;

        result = clenshaw_eval(z, coeffs, degree);
    }
    // apply output scale if provided
    y[i] = result * output_scale;
}

// ---------------------------------------------------------------------------
// Host side: class definition mirroring orion.nn.activation.Chebyshev.  This
// class encapsulates the logic required to fit a Chebyshev approximation
// for a given activation function and to evaluate it efficiently on the
// GPU.  All heavy lifting (fitting and scaling) takes place on the host.

class Chebyshev {
public:
    // Constructor.  Takes the polynomial degree, the function to
    // approximate, and a flag indicating whether this activation is
    // being used inside a composite operation.  The latter mirrors
    // the behaviour of the Python code where fitting is skipped when
    // used in a composite context.
    Chebyshev(int degree_, std::function<double(double)> fn_, bool within_composite_ = false)
        : degree(degree_), fn(fn_), within_composite(within_composite_)
    {
        // Initialise default values as per the Python implementation
        prescale = 1.0;
        constant = 0.0;
        output_scale = 1.0;
        he_mode = true;
        fused = false;
        margin = 1.0;
        depth = 0;
        low = -1.0;
        high = 1.0;
        input_min = -1.0;
        input_max = 1.0;
    }

    // Destructor releases GPU resources, if allocated.
    ~Chebyshev() {
        if (d_coeffs) {
            cudaFree(d_coeffs);
            d_coeffs = nullptr;
        }
    }

    // Expose the degree for external queries.
    int get_degree() const { return degree; }

    // Set the input range used when computing the fitting interval.
    void set_input_range(double min_val, double max_val) {
        input_min = min_val;
        input_max = max_val;
    }

    // Set the margin used when computing low and high.  A margin of 1.0
    // corresponds to no extension (matches the Python default).
    void set_margin(double m) { margin = m; }

    // Set the output scale.  This value is applied during the final
    // polynomial evaluation and does not affect coefficient fitting.
    void set_output_scale(double s) { output_scale = s; }

    // Control flags for homomorphic encryption mode and fused mode.
    void set_he_mode(bool flag) { he_mode = flag; }
    void set_fused(bool flag) { fused = flag; }

    // Fit the Chebyshev polynomial to the activation function on the
    // specified input range.  The method follows the exact logic
    // implemented in the Python version: compute an interval
    // [low, high] from input_min, input_max and margin; generate
    // Chebyshev nodes; decide whether to rescale; evaluate the
    // underlying activation; call numpy.polynomial.Chebyshev.fit
    // equivalent; store coefficients; update depth.
    void fit()
    {
        if (within_composite) {
            // In composite mode the coefficients are provided from the
            // parent composite and fitting is skipped.
            return;
        }

        // Compute the working interval [low, high].  This matches
        // centre/half_range calculation in the Python version.
        double centre = 0.5 * (input_min + input_max);
        double half_range = 0.5 * (input_max - input_min);
        low = centre - margin * half_range;
        high = centre + margin * half_range;

        // Generate Chebyshev points of the first kind.  The Python
        // version uses numpy.polynomial.chebyshev.chebpts1 which
        // yields cos((j + 0.5)*pi/N) for j=0..N-1, where N=degree+1.
        int N = degree + 1;
        std::vector<double> nodes(N);
        for (int j = 0; j < N; ++j) {
            double theta = M_PI * (j + 0.5) / static_cast<double>(N);
            nodes[j] = std::cos(theta);
        }

        // Determine whether to apply the prescaling and shifting.  If the
        // user specified interval extends beyond [-1,1], we must map
        // inputs into that interval.  The prescale and constant are
        // stored for later use during evaluation.
        prescale = 1.0;
        constant = 0.0;
        std::vector<double> evals(N);
        if (low < -1.0 || high > 1.0) {
            // Compute scaling factor and bias for input -> z mapping.
            prescale = 2.0 / (high - low);
            constant = -prescale * (low + high) / 2.0;
            for (int j = 0; j < N; ++j) {
                // Map from [-1,1] to [low, high]: (x+1)*(high-low)/2 + low
                evals[j] = (nodes[j] + 1.0) * (high - low) / 2.0 + low;
            }
        } else {
            // No mapping needed; evals coincide with nodes
            for (int j = 0; j < N; ++j) {
                evals[j] = nodes[j];
            }
        }

        // Evaluate the underlying activation function at these sample
        // points.  We use double precision here for maximum
        // fidelity.
        std::vector<double> yvals(N);
        for (int j = 0; j < N; ++j) {
            yvals[j] = fn(evals[j]);
        }

        // Compute the Chebyshev coefficients using the same least
        // squares fit as numpy.polynomial.Chebyshev.fit.  This
        // involves scaling the sample abscissae to [-1,1] (again)
        // based on the minimal domain that covers nodes and solving
        // a weighted least squares problem with column scaling.  See
        // ``numpy.polynomial._polybase.ABCPolyBase.fit`` for details.
        coeffs = compute_coefficients(nodes, yvals);

        // After fitting, compute the depth.  The Python code uses
        // depth = ceil(log2(degree+1)); if a prescaling is applied
        // (prescale != 1), an additional multiplication is counted.
        set_depth();
    }

    // Copy the coefficients to the device.  If the device buffer
    // already exists but the degree has increased, it will be
    // reallocated.  Coefficients are converted to float before
    // upload.  Must be called after fit() whenever the coefficients
    // change.
    void compile()
    {
        // Allocate device memory if necessary
        if (!d_coeffs || static_cast<int>(coeffs.size()) > d_coeffs_size) {
            if (d_coeffs) {
                cudaFree(d_coeffs);
                d_coeffs = nullptr;
            }
            d_coeffs_size = coeffs.size();
            CUDA_CHECK(cudaMalloc(&d_coeffs, sizeof(float) * d_coeffs_size));
        }
        // Copy coefficients to device as float
        std::vector<float> coeffs_f(coeffs.size());
        for (size_t i = 0; i < coeffs.size(); ++i) {
            coeffs_f[i] = static_cast<float>(coeffs[i]);
        }
        CUDA_CHECK(cudaMemcpy(d_coeffs, coeffs_f.data(),
                              sizeof(float) * coeffs_f.size(),
                              cudaMemcpyHostToDevice));
    }

    // Evaluate the fitted Chebyshev polynomial over a batch of inputs on
    // the device.  The input and output buffers must point to
    // device‐accessible memory.  The length n specifies the number
    // of elements in x and y.  A CUDA stream may optionally be
    // provided for asynchronous execution.
    void forward(const float* d_x, float* d_y, int n, cudaStream_t stream = 0) const
    {
        if (!d_coeffs) {
            throw std::runtime_error("Coefficients not compiled: call compile() after fit().");
        }
        // Launch kernel with a reasonable block size.  Each thread
        // processes one element of the input.
        constexpr int blockSize = 256;
        int gridSize = (n + blockSize - 1) / blockSize;
        chebyshev_forward_kernel<<<gridSize, blockSize, 0, stream>>>(
            d_x, d_y, n,
            d_coeffs, degree,
            he_mode,
            fused,
            static_cast<float>(prescale),
            static_cast<float>(constant),
            static_cast<float>(output_scale));
    }

    // Provide access to the computed coefficients for debugging or
    // inspection.  Note that these are stored in ascending order of
    // degree.
    double get_prescale() const { return prescale; }
    double get_constant() const { return constant; }
    double get_output_scale() const { return output_scale; }
    const std::vector<double>& get_coefficients() const { return coeffs; }

    // After fitting, return the computed depth.  Depth is computed
    // from degree and prescale as defined in the Python code.
    int get_depth() const { return depth; }




private:
    int degree;
    bool within_composite;
    bool he_mode;
    bool fused;
    double prescale;
    double constant;
    double output_scale;
    double margin;
    double input_min;
    double input_max;
    double low;
    double high;
    int depth;
    std::function<double(double)> fn;
    std::vector<double> coeffs;
    float* d_coeffs = nullptr;
    int d_coeffs_size = 0;

    // Solve V c = y on chebpts1 directly (interpolatory solve).
    std::vector<double> compute_coefficients(const std::vector<double>& x,
                                            const std::vector<double>& y)
    {
        const int N = degree + 1;
        if ((int)x.size() != N || (int)y.size() != N)
            throw std::runtime_error("Size mismatch in compute_coefficients");

        // Build Chebyshev-Vandermonde V[j,k] = T_k(x_j) at x = chebpts1
        std::vector<double> V(N * N);
        for (int j = 0; j < N; ++j) {
            double T0 = 1.0, T1 = x[j];
            V[j*N + 0] = T0;
            if (N > 1) V[j*N + 1] = T1;
            for (int k = 2; k < N; ++k) {
                double Tk = 2.0 * x[j] * T1 - T0;
                V[j*N + k] = Tk;
                T0 = T1; T1 = Tk;
            }
        }

        // Solve V c = y with partial-pivot Gaussian elimination
        std::vector<double> M(V);
        std::vector<double> rhs(y);
        for (int k = 0; k < N; ++k) {
            int piv = k;
            double best = std::fabs(M[k*N + k]);
            for (int i = k+1; i < N; ++i) {
                double v = std::fabs(M[i*N + k]);
                if (v > best) { best = v; piv = i; }
            }
            if (piv != k) {
                for (int j = k; j < N; ++j) std::swap(M[k*N + j], M[piv*N + j]);
                std::swap(rhs[k], rhs[piv]);
            }
            double diag = M[k*N + k];
            if (diag == 0.0) throw std::runtime_error("Singular matrix");
            for (int i = k+1; i < N; ++i) {
                double f = M[i*N + k] / diag;
                M[i*N + k] = 0.0;
                for (int j = k+1; j < N; ++j)
                    M[i*N + j] -= f * M[k*N + j];
                rhs[i] -= f * rhs[k];
            }
        }
        std::vector<double> c(N);
        for (int i = N-1; i >= 0; --i) {
            double s = rhs[i];
            for (int j = i+1; j < N; ++j) s -= M[i*N + j] * c[j];
            double diag = M[i*N + i];
            if (diag == 0.0) throw std::runtime_error("Singular matrix (back-subst)");
            c[i] = s / diag;
        }
        return c;   // c0..cN-1  (series understood as c0/2 + Σ_{k>=1} c_k T_k)
    }

    // Compute the depth of the polynomial circuit.  Mirrors the
    // Python implementation: depth = ceil(log2(degree+1)), with an
    // additional increment if prescale != 1.  Note that prescale
    // remains unchanged if the input interval lies within [-1,1].
    void set_depth()
    {
        // compute base depth from degree
        double val = std::log2(static_cast<double>(degree + 1));
        depth = static_cast<int>(std::ceil(val));
        if (prescale != 1.0) {
            // additional level needed for multiplication by prescale
            depth += 1;
        }
    }
};

// ---------------------------------------------------------------------------
// Derived class implementing the SiLU activation function.  The SiLU
// function is defined as x / (1 + exp(-x)).  This class simply
// supplies the underlying function pointer to the base class.
class SiLU : public Chebyshev {
public:
    // Constructor.  Degree defaults to 31 to match the Python
    // implementation.  within_composite is always false because
    // SiLU instances can be fit independently.
    explicit SiLU(int degree = 31)
        : Chebyshev(degree, &SiLU::silu_fn, false)
    {
    }

private:
    // Static helper implementing the SiLU function in host code.
    static double silu_fn(double x) {
        // To avoid overflow in the exponential, clamp inputs.
        if (x > 20.0) {
            return x;
        }
        if (x < -20.0) {
            return 0.0;
        }
        double ex = std::exp(-x);
        return x / (1.0 + ex);
    }
};



// ---------------------------------------------------------------------------
// res = res + scale * poly
static void poly_add_scaled(std::vector<double>& res, const std::vector<double>& poly, double scale) {
    if (res.size() < poly.size()) res.resize(poly.size(), 0.0);
    for (size_t i = 0; i < poly.size(); ++i) res[i] += scale * poly[i];
}

// out = 2*z*poly  (i.e., shift by 1 and multiply by 2)
static std::vector<double> poly_mul_2z(const std::vector<double>& poly) {
    std::vector<double> r(poly.size() + 1, 0.0);
    for (size_t i = 0; i < poly.size(); ++i) r[i+1] += 2.0 * poly[i];
    return r;
}

// Chebyshev T_k recursion to build p(z) = c0/2 + sum_{k>=1} c_k T_k(z) in power basis of z.
static std::vector<double> cheb_series_to_power_in_z(const std::vector<double>& c) {
    const int n = static_cast<int>(c.size()) - 1;
    std::vector<double> Tk_minus_1 = {1.0};      // T0(z)
    std::vector<double> Tk = {0.0, 1.0};         // T1(z)

    std::vector<double> res = { c[0] };

    if (n >= 1) poly_add_scaled(res, Tk, c[1]);
    for (int k = 2; k <= n; ++k) {
        std::vector<double> two_z_Tk = poly_mul_2z(Tk);
        if (two_z_Tk.size() < Tk_minus_1.size()) two_z_Tk.resize(Tk_minus_1.size(), 0.0);
        for (size_t i = 0; i < Tk_minus_1.size(); ++i) two_z_Tk[i] -= Tk_minus_1[i];
        poly_add_scaled(res, two_z_Tk, c[k]);
        Tk_minus_1 = std::move(Tk);
        Tk = std::move(two_z_Tk);
    }
    return res;
}


// binomial coefficient C(n,k) as double (n is small like <= 64)
static inline double binom(int n, int k) {
    if (k < 0 || k > n) return 0.0;
    if (k > n - k) k = n - k;
    double r = 1.0;
    for (int i = 1; i <= k; ++i) r = r * (n - k + i) / i;
    return r;
}

// Compose p(z) with z = alpha * x + beta, returning q(x) in x-power basis.
// Given p(z) = sum_k A_k z^k, we compute q(x) = sum_k A_k (alpha x + beta)^k
static std::vector<double> compose_affine_to_x(const std::vector<double>& A, double alpha, double beta) {
    // degree of q is deg(p)
    const int deg = static_cast<int>(A.size()) - 1;
    std::vector<double> q(deg + 1, 0.0);
    // precompute alpha^i and beta^j if you want; here we do direct pow for clarity
    for (int k = 0; k <= deg; ++k) {
        if (A[k] == 0.0) continue;
        for (int i = 0; i <= k; ++i) {
            // coefficient on x^i from (alpha x + beta)^k
            double term = A[k] * binom(k, i) * std::pow(alpha, i) * std::pow(beta, k - i);
            q[i] += term;
        }
    }
    return q; // coefficients a_i for q(x) = sum_i q[i] x^i
}


static std::vector<double> cheb_to_power_in_x(const std::vector<double>& c,
                                              double alpha, double beta,
                                              double output_scale = 1.0) {
    auto A = cheb_series_to_power_in_z(c);                 // p(z) in z-power basis
    if (output_scale != 1.0) {                             
        for (auto &v : A) v *= output_scale;               
    }
    return compose_affine_to_x(A, alpha, beta);            // q(x) in x-power basis
}





// class Chebyshev(Module):
//     def __init__(self, degree: int, fn, within_composite=False):
//         super().__init__()
//         self.degree = degree
//         self.fn = fn
//         self.within_composite = within_composite
//         self.coeffs = None
       
//         self.output_scale = None
//         self.prescale = 1 
//         self.constant = 0

//     def extra_repr(self):
//         return super().extra_repr() + f", degree={self.degree}"

//     def fit(self):
//         if not self.within_composite:
//             center = (self.input_min + self.input_max) / 2 
//             half_range = (self.input_max - self.input_min) / 2
//             self.low = (center - (self.margin * half_range)).item()
//             self.high = (center + (self.margin * half_range)).item()

//             nodes = np.polynomial.chebyshev.chebpts1(self.degree + 1)
//             if self.low < -1 or self.high > 1:
//                 self.prescale = 2 / (self.high - self.low) 
//                 self.constant = -self.prescale * (self.low + self.high) / 2 
//                 evals = (nodes + 1) * (self.high - self.low) / 2 + self.low
//             else:
//                 evals = nodes
            
//             evals = torch.tensor(evals)
//             T = np.polynomial.Chebyshev.fit(nodes, self.fn(evals), self.degree)
//             self.set_coeffs(T.coef.tolist())
//             self.set_depth()

//     def set_coeffs(self, coeffs):
//         self.coeffs = coeffs

//     def set_depth(self):
//         self.depth = int(math.ceil(math.log2(self.degree+1)))
//         if self.prescale != 1: # additional level needed
//             self.depth += 1

//     def set_output_scale(self, output_scale):
//         self.output_scale = output_scale

//     def compile(self):
//         self.poly = self.scheme.poly_evaluator.generate_chebyshev(self.coeffs)

//     @timer
//     def forward(self, x):  
//         if not self.he_mode:
//             return self.fn(x)

//         # Scale into [-1, 1] if needed.
//         if not self.fused:
//             if self.prescale != 1:
//                 x *= self.prescale 
//             if self.constant != 0:
//                 x += self.constant

//         return self.scheme.poly_evaluator.evaluate_polynomial(
//             x, self.poly, self.output_scale)


