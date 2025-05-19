/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <getopt.h>
#include <cuda_runtime.h>
#include <chrono>
#include <memory>
#include <string>

#include <holoscan/holoscan.hpp>

namespace holoscan::ops {

class ExternalMemorySourceOp : public Operator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS(ExternalMemorySourceOp)

  ExternalMemorySourceOp() = default;

  void setup(OperatorSpec& spec) override {
    spec.output<TensorMap>("out");
    spec.param(size_, "size", "Buffer size", "Size of buffer in bytes", static_cast<int64_t>(1<<20));
    spec.param(async_, "async", "Use cudaMallocAsync", "Allocate asynchronously", false);
    spec.param(copy_, "copy", "Copy mode", "Copy buffer into tensor", false);
    spec.param(allocator_, "allocator", "Allocator", "Tensor allocator", make_resource<UnboundedAllocator>("pool"));
  }

  void compute([[maybe_unused]] InputContext& op_input, OutputContext& op_output, ExecutionContext& context) override {
    TensorMap out_message;
    auto gxf_tensor = std::make_shared<nvidia::gxf::Tensor>();

    nvidia::gxf::Shape shape({static_cast<int32_t>(size_.get())});
    auto dtype = nvidia::gxf::PrimitiveType::kUnsigned8;
    uint64_t bpe = nvidia::gxf::PrimitiveTypeSize(dtype);
    auto strides = nvidia::gxf::ComputeTrivialStrides(shape, bpe);
    nvidia::gxf::MemoryStorageType storage_type = nvidia::gxf::MemoryStorageType::kDevice;

    if (!copy_.get()) {
      if (!async_.get()) {
        auto pointer = std::shared_ptr<void*>(new void*, [](void** p) {
          if (p) {
            cudaFree(*p);
            delete p;
          }
        });
        cudaMalloc(pointer.get(), size_.get());
        gxf_tensor->wrapMemory(shape, dtype, bpe, strides, storage_type, *pointer,
                               [orig = pointer](void*) mutable {
                                 orig.reset();
                                 return nvidia::gxf::Success;
                               });
      } else {
        const std::string stream_name = fmt::format("{}_stream", name_);
        auto maybe_stream = context.allocate_cuda_stream(stream_name);
        if (!maybe_stream) {
          throw std::runtime_error(fmt::format("Failed to allocate CUDA stream: {}", maybe_stream.error().what()));
        }
        auto stream = maybe_stream.value();
        op_output.set_cuda_stream(stream, "out");
        auto pointer = std::shared_ptr<void*>(new void*, [stream](void** p) {
          if (p) {
            cudaFreeAsync(*p, stream);
            delete p;
          }
        });
        cudaMallocAsync(pointer.get(), size_.get(), stream);
        gxf_tensor->wrapMemory(shape, dtype, bpe, strides, storage_type, *pointer,
                               [orig = pointer](void*) mutable {
                                 orig.reset();
                                 return nvidia::gxf::Success;
                               });
      }
    } else {
      void* ptr = nullptr;
      cudaStream_t stream = cudaStreamDefault;
      if (!async_.get()) {
        cudaMalloc(&ptr, size_.get());
      } else {
        const std::string stream_name = fmt::format("{}_stream", name_);
        auto maybe_stream = context.allocate_cuda_stream(stream_name);
        if (!maybe_stream) {
          throw std::runtime_error(fmt::format("Failed to allocate CUDA stream: {}", maybe_stream.error().what()));
        }
        stream = maybe_stream.value();
        op_output.set_cuda_stream(stream, "out");
        cudaMallocAsync(&ptr, size_.get(), stream);
      }

      auto result = gxf_tensor->reshapeCustom(shape, dtype, bpe, strides, storage_type, allocator_.value());
      if (!result) {
        HOLOSCAN_LOG_ERROR("failed to allocate tensor");
      }

      if (!async_.get()) {
        cudaMemcpy(gxf_tensor->pointer(), ptr, size_.get(), cudaMemcpyDeviceToDevice);
        cudaFree(ptr);
      } else {
        cudaMemcpyAsync(gxf_tensor->pointer(), ptr, size_.get(), cudaMemcpyDeviceToDevice, stream);
        cudaFreeAsync(ptr, stream);
      }
    }

    auto maybe_dl_ctx = gxf_tensor->toDLManagedTensorContext();
    if (!maybe_dl_ctx) {
      HOLOSCAN_LOG_ERROR("failed to get DLManagedTensorContext from Tensor");
    }
    std::shared_ptr<Tensor> holoscan_tensor = std::make_shared<Tensor>(maybe_dl_ctx.value());

    out_message.insert({"tensor", holoscan_tensor});
    op_output.emit(out_message, "out");
  }

 private:
  Parameter<int64_t> size_;
  Parameter<bool> async_;
  Parameter<bool> copy_;
  Parameter<std::shared_ptr<Allocator>> allocator_;
};

class BenchmarkRxOp : public Operator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS(BenchmarkRxOp)

  BenchmarkRxOp() = default;

  void setup(OperatorSpec& spec) override {
    spec.input<TensorMap>("in");
    spec.param(expected_, "count", "Count", "Expected messages", static_cast<int64_t>(1));
  }

  void start() override { start_time_ = std::chrono::high_resolution_clock::now(); }

  void stop() override {
    auto end = std::chrono::high_resolution_clock::now();
    double ms = std::chrono::duration<double, std::milli>(end - start_time_).count();
    HOLOSCAN_LOG_INFO("Received %zu messages in %.3f ms", count_, ms);
  }

  void compute(InputContext& op_input, [[maybe_unused]] OutputContext& op_output, [[maybe_unused]] ExecutionContext& context) override {
    auto msg = op_input.receive<TensorMap>("in").value();
    cudaStream_t stream = op_input.receive_cuda_stream("in", false);
    cudaStreamSynchronize(stream);
    (void)msg;
    ++count_;
  }

 private:
  size_t count_ = 0;
  Parameter<int64_t> expected_;
  std::chrono::time_point<std::chrono::high_resolution_clock> start_time_;
};

}  // namespace holoscan::ops

class ExternalMemoryApp : public holoscan::Application {
 public:
  ExternalMemoryApp(size_t size, size_t count, bool async_alloc, bool copy)
      : size_(size), count_(count), async_alloc_(async_alloc), copy_(copy) {}

  void compose() override {
    using namespace holoscan;
    auto tx = make_operator<ops::ExternalMemorySourceOp>(
        "source", make_condition<CountCondition>(count_),
        Arg("size", static_cast<int64_t>(size_)),
        Arg("async", async_alloc_),
        Arg("copy", copy_));
    auto rx = make_operator<ops::BenchmarkRxOp>("sink", Arg("count", static_cast<int64_t>(count_)));
    add_flow(tx, rx);
  }

 private:
  size_t size_;
  size_t count_;
  bool async_alloc_;
  bool copy_;
};

static void print_usage(const char* app) {
  std::cout << "Usage: " << app << " [options]\n"
            << "Options:\n"
            << "  -s, --size <bytes>   buffer size (default: 1048576)\n"
            << "  -n, --count <num>    number of messages (default: 100)\n"
            << "  -a, --async          use cudaMallocAsync\n"
            << "  -c, --copy           copy buffer into tensor\n"
            << "  -h, --help           print this message\n";
}

int main(int argc, char** argv) {
  size_t size = 1 << 20;  // 1MB
  size_t count = 100;
  bool async = false;
  bool copy = false;

  struct option long_options[] = {{"help", no_argument, nullptr, 'h'},
                                  {"size", required_argument, nullptr, 's'},
                                  {"count", required_argument, nullptr, 'n'},
                                  {"async", no_argument, nullptr, 'a'},
                                  {"copy", no_argument, nullptr, 'c'},
                                  {nullptr, 0, nullptr, 0}};

  while (true) {
    int option_index = 0;
    int c = getopt_long(argc, argv, "hs:n:ac", long_options, &option_index);
    if (c == -1) break;
    switch (c) {
      case 'h':
        print_usage(argv[0]);
        return 0;
      case 's':
        size = std::stoul(optarg);
        break;
      case 'n':
        count = std::stoul(optarg);
        break;
      case 'a':
        async = true;
        break;
      case 'c':
        copy = true;
        break;
      default:
        print_usage(argv[0]);
        return 1;
    }
  }

  auto app = holoscan::make_application<ExternalMemoryApp>(size, count, async, copy);
  app->run();

  return 0;
}
