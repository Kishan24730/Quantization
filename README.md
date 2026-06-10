# Quantization
Building Quantization Model from scratch to convert a FP32 model into FP16, Int16, Int8 model with any kind of delegates.

# ONNX to TFLite INT8 Quantization Pipeline 🚀

A robust, production-ready Post-Training Quantization (PTQ) script designed to convert **FP32 ONNX** models into fully optimized **FP16, INT16 and INT8 TFLite** models for edge device deployment (e.g., mobile, IoT, coral, embedded systems).

This wrapper explicitly handles common enterprise environment bottlenecks—such as corporate firewalls blocking network dependencies or dimensional layout mismatches ($NCHW \leftrightarrow NHWC$) during the conversion stream.

---

## ✨ Key Features

* 🔒 **Offline / Firewall Bypass:** Intercepts and patches downstream network verification bugs in `onnx2tf` (resolving `_pickle.UnpicklingError` issues commonly caused by proxy blocks).
* 🔄 **Smart Layout Adaptation:** Dynamically monitors the generated TensorFlow SavedModel graph signature to auto-transpose input shapes seamlessly between **NCHW** and **NHWC**.
* 🎯 **Full Integer Quantization:** Restricts both input and output tensors to explicit `int8` structures for optimal hardware acceleration.
* 📊 **Automated Accuracy Cross-Check:** Validates the newly quantized model by testing it alongside the original ONNX graph to calculate **Mean Squared Error (MSE)** and **Cosine Similarity**.

---

## 🛠️ Prerequisites & Installation

Ensure you are using a clean virtual environment (**Python 3.10+** recommended). Install the locked dependencies:

```bash
pip install onnx2tf tensorflow onnxruntime numpy "protobuf>=5.26" opencv-python
