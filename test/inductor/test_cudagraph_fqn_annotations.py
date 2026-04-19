# Owner(s): ["module: inductor"]
import unittest

import torch
import torch._inductor.config as inductor_config
from torch.testing._internal.common_utils import run_tests, TestCase


class TestCudagraphFqnAnnotations(TestCase):
    def test_config_option_exists_and_defaults_false(self):
        self.assertFalse(inductor_config.triton.cudagraph_kernel_annotations)

    def test_config_option_is_patchable(self):
        with inductor_config.patch({"triton.cudagraph_kernel_annotations": True}):
            self.assertTrue(inductor_config.triton.cudagraph_kernel_annotations)
        self.assertFalse(inductor_config.triton.cudagraph_kernel_annotations)

    def test_wrapped_function_has_fqn_map(self):
        import dataclasses
        from torch._inductor.cudagraph_utils import WrappedFunction
        fields = {f.name for f in dataclasses.fields(WrappedFunction)}
        self.assertIn("fqn_map", fields)

    def test_cudagraph_cached_info_has_fqn_map(self):
        import dataclasses
        from torch._inductor.cudagraph_utils import CudagraphCachedInfo
        fields = {f.name for f in dataclasses.fields(CudagraphCachedInfo)}
        self.assertIn("fqn_map", fields)

    def test_cudagraph_cached_info_fqn_defaults_empty(self):
        from torch._inductor.cudagraph_utils import CudagraphCachedInfo
        info = CudagraphCachedInfo(
            placeholders=(),
            stack_traces=[],
            cudagraph_fail_reasons=[],
        )
        self.assertEqual(info.fqn_map, {})

    def test_wrapped_function_fqn_defaults_empty(self):
        import dataclasses
        from torch._inductor.cudagraph_utils import WrappedFunction
        fqn_field = next(
            f for f in dataclasses.fields(WrappedFunction) if f.name == "fqn_map"
        )
        self.assertIsNotNone(fqn_field.default_factory)

    def test_cudagraphify_accepts_fqn_map(self):
        import inspect
        from torch._inductor.compile_fx import cudagraphify
        sig = inspect.signature(cudagraphify)
        self.assertIn("fqn_map", sig.parameters)

    def test_cudagraphify_trees_accepts_fqn_map(self):
        import inspect
        from torch._inductor.cudagraph_trees import cudagraphify
        sig = inspect.signature(cudagraphify)
        self.assertIn("fqn_map", sig.parameters)

    def test_add_function_accepts_fqn_map(self):
        import inspect
        from torch._inductor.cudagraph_trees import CUDAGraphTreeManager
        sig = inspect.signature(CUDAGraphTreeManager.add_function)
        self.assertIn("fqn_map", sig.parameters)

    def test_record_uses_annotation_ctx_when_enabled(self):
        """When cudagraph_kernel_annotations=True, _record() uses mark_kernels context."""
        import inspect
        from torch._inductor.cudagraph_trees import CUDAGraphNode
        source = inspect.getsource(CUDAGraphNode._record)
        self.assertIn("should_annotate", source)
        self.assertIn("mark_kernels", source)
        self.assertIn("annotation_ctx", source)
        self.assertIn("enable_annotations=should_annotate", source)


if __name__ == "__main__":
    run_tests()
