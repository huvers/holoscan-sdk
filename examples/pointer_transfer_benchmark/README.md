# Pointer Transfer Benchmark

This example measures the cost of passing data between two operators using either
raw pointers or `std::shared_ptr`.

## Usage

Build the SDK and run the example from the build (or install) directory.

```bash
./examples/pointer_transfer_benchmark/cpp/pointer_transfer_benchmark [options]
```

### Options

- `--shared`          Use `std::shared_ptr` (default is raw pointer)
- `--scheduler TYPE`  Scheduler to use (`greedy`, `multi_thread`, `event_based`)
- `--size BYTES`      Size of the payload in bytes (default: 4096)
- `--count N`         Number of messages to send (default: 1000)
- `--track`           Enable data flow tracking metrics
