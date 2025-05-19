# External Memory Example

This example demonstrates how to wrap externally allocated CUDA memory into a Holoscan
`Tensor` so that it can be passed between operators without copying. It also
supports an optional copy mode where the externally allocated buffer is copied
into a Holoscan tensor.

Command line options allow selecting the allocation API (`cudaMalloc` vs
`cudaMallocAsync`), enabling copy mode, choosing the buffer size, and the number
of messages to send.

Due to the lack of GPU support in this environment the performance numbers shown
below are placeholders only.

## Example usage

```bash
./external_memory --size 1048576 --count 1000 --async
```

The `--copy` flag enables copy mode.

## Observed performance (placeholder)

| Buffer size (bytes) | Messages | Mode  | Time (ms) |
|--------------------|---------|-------|-----------|
| 1 MiB              | 1000    | wrap  | TODO      |
| 1 MiB              | 1000    | copy  | TODO      |
| 16 MiB             | 1000    | wrap  | TODO      |
| 16 MiB             | 1000    | copy  | TODO      |

