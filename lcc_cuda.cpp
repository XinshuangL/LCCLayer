#include <torch/extension.h>

at::Tensor CCForward(const at::Tensor& input,
                                 const int N,
                                 const int H,
                                 const int W);

at::Tensor CCBackwardDistance(const at::Tensor& output,
                                 const int L,
                                 const int N,
                                 const int H,
                                 const int W);

at::Tensor forward_cuda(const at::Tensor input) {
  TORCH_CHECK(input.is_cuda(), "input to the LCC layer must be on the GPU device.");
  int N = input.size(0);
  int H = input.size(2);
  int W = input.size(3);
  return CCForward(input, N, H, W);
}

at::Tensor backward_distance_cuda(const at::Tensor output, const int L) {
  TORCH_CHECK(output.is_cuda(), "the output variable for backward must be on the GPU device.");
  int N = output.size(0);
  int H = output.size(2);
  int W = output.size(3);
  return CCBackwardDistance(output, L, N, H, W);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward_cuda", &forward_cuda,
        "CC forward");
  m.def("backward_distance_cuda", &backward_distance_cuda,
        "CC backward");
}
