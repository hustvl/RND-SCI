#include "csrc/roll_optim.h"

namespace hsi {

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("roll_optim", &roll_optim, "roll_optim");
}

}  // namespace hsi
