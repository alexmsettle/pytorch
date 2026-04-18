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
        # fqn_map field has default_factory=dict so default value is {}
        fqn_field = next(
            f for f in dataclasses.fields(WrappedFunction) if f.name == "fqn_map"
        )
        self.assertIsNotNone(fqn_field.default_factory)


if __name__ == "__main__":
    run_tests()
