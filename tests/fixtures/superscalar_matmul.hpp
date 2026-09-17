#ifndef MATMUL_FIXTURE_HPP
#define MATMUL_FIXTURE_HPP
using namespace pto;
template <int tM, int tN, int tK>
__attribute__((noinline)) void matmul_test(__half *dst, __half *src0, __half *src1,
                                           float *scratch, int gM, int gN, int gK,
                                           int lda, int ldb, int ldc) {
  using gm_shapeA = global_tensor<__half, RowMajor<-1, -1>>;
  using tile_shapeA = CubeTileM16<__half, tM, tK>;
  using tile_shapeACC = CubeAccumulatorM16<float, tM, tN>;
  for (int i = 0; i < gM; i += tM) {
    for (int j = 0; j < gN; j += tN) {
      tile_shapeACC tACC;
      for (int k = 0; k < gK; k += tK) {
        gm_shapeA gA(src0 + i * lda + k, gM, lda);
        tile_shapeA tA;
        TLOAD_CUBE(tA, gA);
        TMATMUL(tACC, tA, tA);
      }
      TSTORE_CUBE(scratch, tACC);
    }
  }
}
#endif
