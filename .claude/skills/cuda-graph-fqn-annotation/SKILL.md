---
name: cuda-graph-fqn-annotation
description: Reference for the CUDA graph FQN annotation feature on the cuda_graph_markers branch. Use when diagnosing fqn_trace logs, understanding the nn_module_stack pipeline, or debugging mark_kernels output in generated code.
---

# CUDA Graph FQN Annotation

Binds `nn.Module` layer names to CUDA graph kernels as NVTX/profiler ranges via `_graph_annotations.mark_kernels(fqn)`. Implemented on the `cuda_graph_markers` branch (fork: `alexmsettle/pytorch`).

Controlled by `torch._inductor.config.triton.cudagraph_kernel_annotations`.

## nn_module_stack pipeline

The FQN flows from FX graph metadata all the way to the generated Python wrapper code.

### Stage 1 — FQN extraction (`graph_view.py:get_fused_kernel_module_fqn`)

Called with a list of `SchedulerNode`s. For each node reads `snode.node.get_origin_node()` (set via `IRNode._current_primary_node` during `graph.py:run_node`), reads `node.meta["nn_module_stack"]`, applies `_clean_stack_name` to the innermost entry, and joins unique names with `" + "`.

Log lines:
```
[fqn_trace] get_fused_kernel_module_fqn: snode=<name> origin_node=<fx_node>
[fqn_trace] get_fused_kernel_module_fqn: result=<fqn>   # None if no stack found
```

### Stage 2a — Triton kernel annotation (`wrapper.py:generate_kernel_call`)

`simd.py:codegen_node` sets `V.graph._current_kernel_module_fqn` before calling `_codegen_nodes`. `generate_kernel_call` reads it and emits `AnnotatedKernelCallLine` instead of a bare `KernelCallLine`.

Log lines:
```
[fqn_trace] generate_kernel_call: kernel=<name> module_fqn=<fqn> cudagraph_kernel_annotations=True
[fqn_trace] generate_kernel_call: emitting AnnotatedKernelCallLine for <name>
[fqn_trace] AnnotatedKernelCallLine.codegen: writing mark_kernels('<fqn>')
```

### Stage 2b — Extern kernel annotation (`scheduler.py:ExternKernelSchedulerNode.codegen`)

Calls `get_fused_kernel_module_fqn([self])` directly. Captures `wrapper.lines` before and after `self.node.codegen(wrapper)`, wraps captured lines in `AnnotatedExternKernelBlock`, substitutes back.

Log lines:
```
[fqn_trace] ExternKernelSchedulerNode.codegen: module_fqn=<fqn> for <node_name>
[fqn_trace] AnnotatedExternKernelBlock.codegen: writing mark_kernels('<fqn>')
```

### Stage 3 — CUDA graph registration (`cudagraph_trees.py`)

```
[fqn_trace] add_function: id=<id>, cudagraph_kernel_annotations=True
[fqn_trace] _record: graph=<id>, cudagraph_kernel_annotations=True
```

## Expected output_code.py

Triton kernel:
```python
with _graph_annotations.mark_kernels('conv'):
    triton_poi_fused_convolution_0.run(...)
```

Extern kernel:
```python
with _graph_annotations.mark_kernels('conv'):
    # Source Nodes: [conv2d], Original ATen: [aten.convolution]
    # Provenance debug handles: ...
    extern_kernels.convolution(primals_3, primals_1, ...)
```

## How origin_node gets set on every IR buffer

**Problem solved (2026-05-03):** `assign_origin_node` only covers primary outputs of `run_node`. Counter-named buffers (e.g. `buf0`) had `origin_node = None`, causing missing FQN annotations for some fused ops.

**Fix:** Added `IRNode._current_primary_node` class variable (non-accumulating) set to the FX node `n` in `graph.py:run_node` via `current_primary_node(n)` context manager. `IRNode.__post_init__` sets `origin_node = _current_primary_node` for every IR node at creation — so ALL buffers created during `run_node(n)` get `origin_node = n`.

`StorageBox.realize()` already reads `origin_node` from the inner `Pointwise`/`Reduction` and copies it to the new `ComputedBuffer`. This means inline realization of a buffer from `run_node(B)` during `run_node(A)` correctly preserves `origin_node = B_fx_node` — no forward cascading.

Key invariant: `origin_node` on a `ComputedBuffer` = the FX node whose `run_node` call created (or last realized) the inner computation.

## Diagnosing failures

| Symptom | Likely cause |
|---|---|
| `result=None` in `get_fused_kernel_module_fqn` | `nn_module_stack` absent on FX nodes — check if `torch.compile` traced through the module |
| `module_fqn=None` in `generate_kernel_call` | `simd.py` didn't set `_current_kernel_module_fqn` — check config flag |
| `ExternKernelSchedulerNode.codegen` log absent | `cudagraph_kernel_annotations=False` or `cpp_wrapper=True` |
| `AnnotatedExternKernelBlock.codegen` absent | Block created but `wrapper.lines` not flushed via normal path |
| `mark_kernels` missing from `output_code.py` | FQN resolved to `None`; check stage 1 logs |
| FQN shows cascading upstream ops | `origins` used instead of `origin_node` — check `get_fused_kernel_module_fqn` |

## Key types

- `AnnotatedKernelCallLine(inner: KernelCallLine, module_fqn: str)` — wraps a single Triton call
- `AnnotatedExternKernelBlock(inner_lines: list[Any], module_fqn: str)` — wraps multi-line extern kernel block (comment + alloc + call)

Both live in `torch/_inductor/codegen/wrapper.py`.

## Enabling annotations at runtime

```python
import torch
with torch.cuda.graph(enable_annotations=True) as g:
    ...
```

`enable_annotations=True` calls `_graph_annotations.enable_annotations()` on enter and `resolve_pending_annotations()` + `remap_to_exec_graph()` on exit.
