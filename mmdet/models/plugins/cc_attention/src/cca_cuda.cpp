#include <torch/extension.h>
#include <cmath>
#include <vector>
#include <THC/THC.h>

int CAForwardLaucher(const int N, const int C,
                     const int H, const int W,
                     const at::Tensor t, const at::Tensor f,
                     at::Tensor weight);

int CABackwardLaucher(const int N, const int C,
                      const int H, const int W,
                      const at::Tensor dw, const at::Tensor t, const at::Tensor f,
                      at::Tensor dt, at::Tensor df);

int CAMapForwardLaucher(const int N, const int C, const int H, const int W,
                        const at::Tensor weight, const at::Tensor g,
                        at::Tensor out);

int CAMapBackwardLaucher(const int N, const int C, const int H, const int W,
                         const at::Tensor dout, const at::Tensor weight, const at::Tensor g,
                         at::Tensor dw, at::Tensor dg);

#define CHECK_CUDA(x) AT_CHECK(x.type().is_cuda(), #x, " must be a CUDAtensor ")
#define CHECK_CONTIGUOUS(x) \
  AT_CHECK(x.is_contiguous(), #x, " must be contiguous ")
#define CHECK_INPUT(x) \
  CHECK_CUDA(x);       \
  CHECK_CONTIGUOUS(x)


int ca_forward_cuda(at::Tensor t, at::Tensor f, at::Tensor weight){
    CHECK_INPUT(t);
    CHECK_INPUT(f);
    CHECK_INPUT(weight);

    int N = t.size(0);
    int C = t.size(1);
    int H = t.size(2);
    int W = t.size(3);

    CAForwardLaucher(N, C, H, W, t, f, weight);

    return 1;
}

int ca_backward_cuda(at::Tensor dw, at::Tensor t, at::Tensor f, at::Tensor dt, at::Tensor df) {
    CHECK_INPUT(dw);
    CHECK_INPUT(t);
    CHECK_INPUT(f);
    CHECK_INPUT(dt);
    CHECK_INPUT(df);

    int N = t.size(0);
    int C = t.size(1);
    int H = t.size(2);
    int W = t.size(3);

    CABackwardLaucher(N, C, H, W, dw, t, f, dt, df);
    return 1;
}

int ca_map_forward_cuda(at::Tensor weight, at::Tensor g, at::Tensor out){
    CHECK_INPUT(weight);
    CHECK_INPUT(g);
    CHECK_INPUT(out);

    int N = g.size(0);
    int C = g.size(1);
    int H = g.size(2);
    int W = g.size(3);

    CAMapForwardLaucher(N, C, H, W, weight, g, out);
    return 1;
}

int ca_map_backward_cuda(at::Tensor dout, at::Tensor weight, at::Tensor g,
                         at::Tensor dw,  at::Tensor dg) {
    CHECK_INPUT(dout);
    CHECK_INPUT(weight);
    CHECK_INPUT(g);
    CHECK_INPUT(dw);
    CHECK_INPUT(dg);

    int N = dout.size(0);
    int C = dout.size(1);
    int H = dout.size(2);
    int W = dout.size(3);

    CAMapBackwardLaucher(N, C, H, W, dout, weight, g, dw, dg);
    return 1;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("ca_forward", &ca_forward_cuda, "CA forward (CUDA)");
    m.def("ca_backward", &ca_backward_cuda, "CA backward (CUDA)");
    m.def("ca_map_forward", &ca_map_forward_cuda, "CA Map forward (CUDA)");
    m.def("ca_map_backward", &ca_map_backward_cuda, "CA Map backward (CUDA)");
}