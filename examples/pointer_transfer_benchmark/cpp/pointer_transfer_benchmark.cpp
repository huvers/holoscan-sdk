#include <chrono>
#include <iostream>
#include <memory>
#include <vector>
#include <algorithm>
#include "holoscan/holoscan.hpp"

using holoscan::Arg;
using holoscan::ArgList;
using holoscan::Condition;
using holoscan::CountCondition;
using holoscan::Operator;
using holoscan::OperatorSpec;
using holoscan::Parameter;
using holoscan::InputContext;
using holoscan::OutputContext;
using holoscan::ExecutionContext;

struct Payload {
  explicit Payload(size_t size) : data(size) {}
  std::vector<uint8_t> data;
};

namespace ops {
class TxRawOp : public Operator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS(TxRawOp)
  TxRawOp() = default;
  void setup(OperatorSpec& spec) override {
    spec.output<Payload*>("out");
    spec.param(size_, "data_size", "Payload size", "Size of the payload", 4096L);
  }
  void start() override { payload_ = new Payload(size_.get()); }
  void stop() override { delete payload_; }
  void compute(InputContext&, OutputContext& out, ExecutionContext&) override {
    out.emit(payload_, "out");
  }
 private:
  Parameter<int64_t> size_;
  Payload* payload_ = nullptr;
};

class RxRawOp : public Operator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS(RxRawOp)
  void setup(OperatorSpec& spec) override { spec.input<Payload*>("in"); }
  void compute(InputContext& in, OutputContext&, ExecutionContext&) override {
    auto ptr = in.receive<Payload*>("in").value();
    volatile uint8_t v = ptr->data[0];
    (void)v;
  }
};

class TxSharedOp : public Operator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS(TxSharedOp)
  void setup(OperatorSpec& spec) override {
    spec.output<std::shared_ptr<Payload>>("out");
    spec.param(size_, "data_size", "Payload size", "Size of the payload", 4096L);
  }
  void start() override { payload_ = std::make_shared<Payload>(size_.get()); }
  void compute(InputContext&, OutputContext& out, ExecutionContext&) override {
    out.emit(payload_, "out");
  }
 private:
  Parameter<int64_t> size_;
  std::shared_ptr<Payload> payload_;
};

class RxSharedOp : public Operator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS(RxSharedOp)
  void setup(OperatorSpec& spec) override { spec.input<std::shared_ptr<Payload>>("in"); }
  void compute(InputContext& in, OutputContext&, ExecutionContext&) override {
    auto ptr = in.receive<std::shared_ptr<Payload>>("in").value();
    volatile uint8_t v = ptr->data[0];
    (void)v;
  }
};
}  // namespace ops

std::optional<bool> get_boolean_arg(std::vector<std::string> args, const std::string& name) {
  if (std::find(args.begin(), args.end(), name) != std::end(args)) { return true; }
  return {};
}
std::optional<int64_t> get_int64_arg(std::vector<std::string> args, const std::string& name) {
  auto loc = std::find(args.begin(), args.end(), name);
  if ((loc != std::end(args)) && (++loc != std::end(args))) {
    try { return std::stoll(*loc); } catch (...) { return {}; }
  }
  return {};
}
std::optional<std::string> get_str_arg(std::vector<std::string> args, const std::string& name) {
  auto loc = std::find(args.begin(), args.end(), name);
  if ((loc != std::end(args)) && (++loc != std::end(args))) { return *loc; }
  return {};
}

class PointerTransferApp : public holoscan::Application {
 public:
  PointerTransferApp(bool use_shared, int64_t count, int64_t size)
      : use_shared_(use_shared), count_(count), size_(size) {}
  void compose() override {
    using namespace holoscan;
    if (use_shared_) {
      auto tx = make_operator<ops::TxSharedOp>("tx", Arg("data_size", size_),
                                              make_condition<CountCondition>(count_));
      auto rx = make_operator<ops::RxSharedOp>("rx");
      add_flow(tx, rx);
    } else {
      auto tx = make_operator<ops::TxRawOp>("tx", Arg("data_size", size_),
                                           make_condition<CountCondition>(count_));
      auto rx = make_operator<ops::RxRawOp>("rx");
      add_flow(tx, rx);
    }
  }
 private:
  bool use_shared_;
  int64_t count_;
  int64_t size_;
};

int main(int argc, char** argv) {
  auto app = holoscan::make_application<PointerTransferApp>(false, 1000, 4096);
  std::vector<std::string>& args = app->argv();
  bool use_shared = get_boolean_arg(args, "--shared").value_or(false);
  int64_t count = get_int64_arg(args, "--count").value_or(1000);
  int64_t size = get_int64_arg(args, "--size").value_or(4096);
  std::string scheduler = get_str_arg(args, "--scheduler").value_or("greedy");
  bool tracking = get_boolean_arg(args, "--track").value_or(false);

  app = holoscan::make_application<PointerTransferApp>(use_shared, count, size);

  if (scheduler == "multi_thread") {
    app->scheduler(app->make_scheduler<holoscan::MultiThreadScheduler>("mts"));
  } else if (scheduler == "event_based") {
    app->scheduler(app->make_scheduler<holoscan::EventBasedScheduler>("ebs"));
  } else if (scheduler == "greedy") {
    app->scheduler(app->make_scheduler<holoscan::GreedyScheduler>("greedy"));
  }

  holoscan::DataFlowTracker* tracker = nullptr;
  if (tracking) { tracker = &app->track(); }

  app->run();

  if (tracker) { tracker->print(); }

  return 0;
}
