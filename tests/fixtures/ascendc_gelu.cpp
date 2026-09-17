# NPU usability fixtures: AscendC-style kernel
#include "kernel_operator.h"
using namespace AscendC;

__aicore__ inline void gelu_kernel(GM_ADDR x, GM_ADDR y, int n) {
  TPipe pipe;
  TQue<QuePosition::VECIN, 1> inQueue;
  TQue<QuePosition::VECOUT, 1> outQueue;
  GlobalTensor<half> gx;
  gx.SetGlobalBuffer((__gm__ half*)x);
  LocalTensor<half> lx = inQueue.AllocTensor<half>();
  DataCopy(lx, gx, n);
  inQueue.EnQue(lx);
  SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
  WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
  LocalTensor<half> ly = outQueue.AllocTensor<half>();
  Add(ly, lx, lx, n);
  Mul(ly, ly, ly, n);
  DataCopy(gx, ly, n);
  outQueue.FreeTensor(ly);
}
