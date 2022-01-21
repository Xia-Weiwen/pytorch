#include <ATen/ATen.h>
#include <ATen/core/op_registration/op_registration.h>
#include <ATen/native/quantized/cpu/fbgemm_utils.h>
#include <ATen/native/quantized/cpu/packed_params.h>
#include <ATen/native/quantized/cpu/qnnpack_utils.h>
#include <ATen/native/quantized/cpu/onednn_utils.h>
#include <caffe2/utils/threadpool/pthreadpool-cpp.h>
#include <torch/custom_class.h>
#include <torch/library.h>

#ifdef USE_FBGEMM
at::Tensor quantized_bmm_fbgemm(
    at::Tensor X,
    at::Tensor Y,
    double output_scale,
    int64_t output_zero_point) {
  TORCH_CHECK(
      fbgemm::fbgemmSupportedCPU(), "Your CPU does not support FBGEMM.");
  TORCH_CHECK(
      X.dim() == 3 && Y.dim() == 3,
      "quantized::bmm() (FBGEMM): Input tensor ranks should both be 3 ",
      "but got", X.dim(), " and ", Y.dim());
  // Reference implementation: dequantize - bmm - quantize
  // Inputs
  auto dqx = X.dequantize();
  auto dqy = Y.dequantize();
  // Output
  auto z = at::bmm(dqx, dqy);
  return at::quantize_per_tensor(
      z, output_scale, output_zero_point, c10::kQUInt8);
}
#endif // USE_FBGEMM

#ifdef USE_PYTORCH_QNNPACK
at::Tensor quantized_bmm_qnnpack(
    at::Tensor X,
    at::Tensor Y,
    double output_scale,
    int64_t output_zero_point) {
  TORCH_CHECK(
      X.dim() == 3 && Y.dim() == 3,
      "quantized::bmm() (QNNPACK): Input tensor ranks should both be 3 ",
      "but got", X.dim(), " and ", Y.dim());
  // Reference implementation: dequantize - bmm - quantize
  // Inputs
  auto dqx = X.dequantize();
  auto dqy = Y.dequantize();
  // Output
  auto z = at::bmm(dqx, dqy);
  return at::quantize_per_tensor(
      z, output_scale, output_zero_point, c10::kQUInt8);
}
#endif // USE_PYTORCH_QNNPACK

#if AT_MKLDNN_ENABLED()
at::Tensor quantized_bmm_onednn(
    at::Tensor X,
    at::Tensor Y,
    double output_scale,
    int64_t output_zero_point) {
  TORCH_CHECK(
      X.dim() == 3 && Y.dim() == 3,
      "quantized::bmm() (ONEDNN): Input tensor ranks should both be 3 ",
      "but got", X.dim(), " and ", Y.dim());

  TORCH_CHECK(X.scalar_type() == c10::ScalarType::QUInt8 &&
      Y.scalar_type() == c10::ScalarType::QUInt8,
      "quantized::bmm() (ONEDNN): data type of inputs should be QUint8.");

  auto B = X.size(0), M = X.size(1), K = X.size(2), N = Y.size(2);
  TORCH_CHECK(B == Y.size(0) && K == Y.size(1),
      "quantized::bmm() (ONEDNN): Shapes of inputs mismatch, got (",
      X.size(0), " ", X.size(1), " ", X.size(2), ") and (",
      Y.size(0), " ", Y.size(1), " ", Y.size(2), ")");

  // src_0: dtype = u8
  auto x_dims = X.sizes().vec();
  auto x_data_type = dnnl::memory::data_type::u8;
  auto x_desc = ideep::tensor::desc(x_dims, x_data_type);
  ideep::tensor x(x_desc, X.data_ptr<c10::quint8>());

  // src_1: original dtype = u8
  auto y_dims = Y.sizes().vec();
  auto y_data_type = dnnl::memory::data_type::u8;
  auto y_u8_desc = ideep::tensor::desc(y_dims, y_data_type);
  ideep::tensor y_u8(y_u8_desc, Y.data_ptr<c10::quint8>());

  // Scales & zero points
  // Scales of ONEDNN and PyTorch are reciprocal
  const ideep::scale_t& x_scales = ideep::scale_t(1, 1.0/X.q_scale());
  const ideep::scale_t& y_scales = ideep::scale_t(1, 1.0/Y.q_scale());
  const ideep::scale_t& z_scales = ideep::scale_t(1, 1.0/output_scale);
  const ideep::zero_point_t& x_zero_point = ideep::zero_point_t(1, X.q_zero_point());
  const ideep::zero_point_t& y_zero_point = ideep::zero_point_t(1, Y.q_zero_point() - 128);
  const ideep::zero_point_t& z_zero_point = ideep::zero_point_t(1, output_zero_point);

  // Reorder src_1 to s8
  auto y_s8_desc = y_u8.get_desc().to_type(dnnl::memory::data_type::s8);
  ideep::tensor y_s8(y_s8_desc);
  ideep::attr_t reorder_attr;
  reorder_attr.set_zero_points(DNNL_ARG_DST, 0, ideep::zero_point_t(1, -128));
  y_u8.reorder_to(y_s8, reorder_attr);

  // Allocate output Tensor
  auto z_dims = {B, M, N};
  at::Tensor output = at::_empty_affine_quantized(
      z_dims,
      at::device(c10::kCPU).dtype(c10::kQUInt8),
      output_scale,
      output_zero_point);
  if (output.numel() == 0) {
    return output;
  }
  ideep::tensor::desc z_desc = {
      z_dims, ideep::tensor::data_type::u8,
      {output.strides().cbegin(), output.strides().cend()}};
  ideep::tensor z(z_desc, output.data_ptr());

  // Compute
  ideep::scale_t op_scales(1); // scales size = 1
  op_scales[0] = z_scales[0] / (x_scales[0] * y_scales[0]);
  ideep::attr_t op_attr;
  op_attr.set_output_scales(ideep::utils::op_scale_mask(1), op_scales);
  op_attr.set_zero_points(DNNL_ARG_SRC,
                          ideep::utils::tensor_zp_mask(1), // zp size = 1
                          x_zero_point);
  op_attr.set_zero_points(DNNL_ARG_WEIGHTS,
                          ideep::utils::tensor_zp_mask(1),
                          y_zero_point);
  op_attr.set_zero_points(DNNL_ARG_DST,
                          ideep::utils::tensor_zp_mask(1),
                          z_zero_point);
  
  auto key = ideep::utils::create_key(
      x_desc,
      y_s8_desc,
      z_desc,
      op_attr,
      omp_get_max_threads());
  auto pd = ideep::utils::computation_cache<dnnl::matmul::primitive_desc>
            ::fetch_or_create(key, [&]() {
      return dnnl::matmul::primitive_desc(
          {x_desc, y_s8_desc, z_desc}, op_attr, ideep::engine::cpu_engine());
  });
  auto expected_x = x.reorder_if_differ_in(pd.src_desc());
  auto expected_y = y_s8.reorder_if_differ_in(pd.weights_desc());
  dnnl::matmul(pd).execute(ideep::stream::default_stream(),
                           {{DNNL_ARG_SRC, expected_x},
                            {DNNL_ARG_WEIGHTS, expected_y},
                            {DNNL_ARG_DST, z}});
  return output;
}
#endif // #if AT_MKLDNN_ENABLED()

namespace at {
namespace native {
namespace {

class QBmmInt8 final {
 public:
  static at::Tensor run(
      at::Tensor X,
      at::Tensor Y,
      double output_scale,
      int64_t output_zero_point) {
    auto& ctx = at::globalContext();
#ifdef USE_FBGEMM
    if (ctx.qEngine() == at::QEngine::FBGEMM) {
      return quantized_bmm_fbgemm(X, Y, output_scale, output_zero_point);
    }
#endif // USE_FBGEMM
#ifdef USE_PYTORCH_QNNPACK
    if (ctx.qEngine() == at::QEngine::QNNPACK) {
      return quantized_bmm_qnnpack(X, Y, output_scale, output_zero_point);
    }
#endif // USE_PYTORCH_QNNPACK
#if AT_MKLDNN_ENABLED()
    if (ctx.qEngine() == at::QEngine::ONEDNN) {
      return quantized_bmm_onednn(X, Y, output_scale, output_zero_point);
    }
#endif // #if AT_MKLDNN_ENABLED()
    TORCH_CHECK(
        false,
        "Didn't find engine for operation quantized::bmm ",
        toString(ctx.qEngine()));
  }
};

TORCH_LIBRARY_IMPL(quantized, QuantizedCPU, m) {
  m.impl(TORCH_SELECTIVE_NAME("quantized::bmm"), TORCH_FN(QBmmInt8::run));
}

} // namespace
} // namespace native
} // namespace at
