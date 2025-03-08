
#include <torch/extension.h>
#include <vector>
#include <string.h>
#include <cstdlib>
#include <map>
#include <chrono>

void matTrans(torch::Tensor AT, torch::Tensor A);

torch::Tensor transpose(torch::Tensor A) {
  torch::Tensor AT = torch::zeros_like(A, torch::TensorOptions().device(A.device()));
  matTrans(AT, A);
  return AT;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  // example:
  m.def("naive_transpose", &transpose, "naive transpose");
  // below are the functions you need to implement and compare
  // m.def("naive_attention", &naiveAttention, "naive attention");
  // m.def("fused_attention", &fusedAttention, "fused attention");
  // m.def("tc_fused_attention", &tcFusedAttention, "fused attention with tensor cores");
  // m.def("sparse_tc_fused_attention", &sparseTcFusedAttention, "sparse fused attention with tensor cores");
  // add more here if you have more variants to test
}
