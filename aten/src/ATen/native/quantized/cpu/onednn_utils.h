#pragma once

#include <ATen/Config.h>
#if AT_MKLDNN_ENABLED()
#include <ATen/Tensor.h>
#include <ATen/native/quantized/packed_params.h>
#include <ATen/native/mkldnn/MKLDNNCommon.h>
#include <ATen/native/mkldnn/Utils.h>

// Base class of primitive cache. Only support conv for now.
struct PrimitiveCache {

  bool is_valid = false;
  double input_scale;
  int64_t input_zero_point;
  std::vector<int64_t> input_shape;
  double output_scale;
  int64_t output_zero_point;

  inline bool valid() {
    return this->is_valid;
  }

  bool hit(
      double input_scale,
      int64_t input_zero_point,
      const std::vector<int64_t>& input_shape,
      double output_scale,
      int64_t output_zero_point) {
    return valid() &&
           this->input_scale == input_scale &&
           this->input_zero_point == input_zero_point &&
           this->input_shape == input_shape &&
           this->output_scale == output_scale &&
           this->output_zero_point == output_zero_point;
  }
};

using Conv = dnnl::convolution_forward;
using ConvParams = ideep::convolution_forward_params;
struct ConvPrimitiveCache : PrimitiveCache {

  ConvParams conv_params;
  Conv primitive;
  ideep::tensor input_zp_tensor;

  inline void set(
      double input_scale,
      int64_t input_zero_point,
      const std::vector<int64_t>& input_shape,
      double output_scale,
      int64_t output_zero_point,
      ideep::convolution_forward_params conv_params) {
    this->input_scale = input_scale;
    this->input_zero_point = input_zero_point;
    this->input_shape = input_shape;
    this->output_scale = output_scale;
    this->output_zero_point = output_zero_point;
    this->conv_params = conv_params;
    this->primitive = Conv(this->conv_params.pd);
    // Construct tensor of input zero point
    ideep::tensor::desc input_zp_desc = {{1}, ideep::data_type::s32, {1}};
    auto aengine = ideep::engine(this->conv_params.pd.get_engine().get_kind());
    this->input_zp_tensor.init(input_zp_desc, aengine);
    auto zp_data_ptr = reinterpret_cast<int32_t *>(this->input_zp_tensor.get_data_handle());
    zp_data_ptr[0] = this->input_zero_point;
    this->is_valid = true;
  }

  inline ConvParams& get_conv_params() {
    return conv_params;
  }

  inline Conv& get_conv_primitive() {
    return primitive;
  }

  inline ideep::tensor& get_src_zp_tensor() {
    return input_zp_tensor;
  }
};

struct PackedLinearWeightsOnednn : public LinearPackedParamsBase {
  PackedLinearWeightsOnednn(
      std::unique_ptr<ideep::tensor> weight,
      c10::optional<ideep::tensor> bias,
      at::Tensor orig_weight,
      c10::optional<at::Tensor> orig_bias)
      : weight_(std::move(weight)),
        bias_(std::move(bias)),
        orig_weight_(std::move(orig_weight)),
        orig_bias_(std::move(orig_bias)) {}
  std::unique_ptr<ideep::tensor> weight_;
  c10::optional<ideep::tensor> bias_;
  at::Tensor orig_weight_;
  c10::optional<at::Tensor> orig_bias_;

  at::Tensor apply(
      at::Tensor input,
      double output_scale,
      int64_t output_zero_point) override;
  at::Tensor apply_relu(
      at::Tensor input,
      double output_scale,
      int64_t output_zero_point) override;

  at::Tensor apply_dynamic(at::Tensor input, bool reduce_range=false) override;
  at::Tensor apply_dynamic_relu(at::Tensor input, bool reduce_range=false) override;

  std::tuple<at::Tensor, c10::optional<at::Tensor>> unpack() override;

  c10::optional<at::Tensor> bias() override {
    return orig_bias_;
  }

  static c10::intrusive_ptr<LinearPackedParamsBase> prepack(
      at::Tensor weight,
      c10::optional<at::Tensor> bias);

 private:
  template <bool ReluFused>
  at::Tensor apply_impl(
      at::Tensor input,
      double output_scale,
      int64_t output_zero_point);

  template <bool ReluFused>
  at::Tensor apply_dynamic_impl(at::Tensor input, bool reduce_range=false);
};

template <int kSpatialDim = 2>
struct PackedConvWeightsOnednn : public ConvPackedParamsBase<kSpatialDim> {
  PackedConvWeightsOnednn(
      std::unique_ptr<ideep::tensor> weight,
      c10::optional<ideep::tensor> bias,
      at::Tensor orig_weight,
      c10::optional<at::Tensor> orig_bias,
      torch::List<int64_t> stride,
      torch::List<int64_t> padding,
      torch::List<int64_t> output_padding,
      torch::List<int64_t> dilation,
      int64_t groups,
      uint8_t transpose)
    : weight_(std::move(weight)),
    bias_(std::move(bias)),
    orig_weight_(std::move(orig_weight)),
    orig_bias_(std::move(orig_bias)),
    stride_(std::move(stride)),
    padding_(std::move(padding)),
    output_padding_(std::move(output_padding)),
    dilation_(std::move(dilation)),
    groups_(groups),
    transpose_(transpose) {}

  std::unique_ptr<ideep::tensor> weight_;
  c10::optional<ideep::tensor> bias_;
  at::Tensor orig_weight_;
  c10::optional<at::Tensor> orig_bias_;
  torch::List<int64_t> stride_;
  torch::List<int64_t> padding_;
  torch::List<int64_t> output_padding_;
  torch::List<int64_t> dilation_;
  int64_t groups_;
  uint8_t transpose_;

  at::Tensor apply(
      const at::Tensor& input,
      double output_scale,
      int64_t output_zero_point) override;

  at::Tensor apply_relu(
      const at::Tensor& input,
      double output_scale,
      int64_t output_zero_point) override;

  at::Tensor apply_dynamic(
      const at::Tensor& input,
      bool reduce_range) override;

  std::tuple<at::Tensor, c10::optional<at::Tensor>> unpack() override;

  static c10::intrusive_ptr<ConvPackedParamsBase<kSpatialDim>> prepack(
      at::Tensor weight,
      c10::optional<at::Tensor> bias,
      torch::List<int64_t> stride,
      torch::List<int64_t> padding,
      torch::List<int64_t> output_padding,
      torch::List<int64_t> dilation,
      int64_t groups,
      bool transpose);

  torch::List<int64_t> stride() const override {
    return stride_;
  }

  torch::List<int64_t> padding() const override {
    return padding_;
  }

  torch::List<int64_t> output_padding() const override {
    return output_padding_;
  }

  torch::List<int64_t> dilation() const override {
    return dilation_;
  }

  int64_t groups() const override {
    return groups_;
  }

  bool transpose() const override {
    return (bool)transpose_;
  }

 private:
  ConvPrimitiveCache conv_prim_cache;

  template <bool ReluFused>
  at::Tensor apply_impl(
      const at::Tensor& input,
      double output_scale,
      int64_t output_zero_point);

  inline ConvPrimitiveCache& get_cache() {
    assert(!transpose());
    return conv_prim_cache;
  }
};

#endif // #if AT_MKLDNN_ENABLED()
