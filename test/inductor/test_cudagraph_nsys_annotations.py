# Owner(s): ["module: inductor"]
import json
import os
import sqlite3
import tempfile
import unittest

from torch.testing._internal.common_utils import run_tests, TestCase


def _make_nsys_db(path: str, kernel_rows: list[tuple]) -> None:
    conn = sqlite3.connect(path)
    conn.execute("""
        CREATE TABLE CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL (
            start INTEGER, end INTEGER, deviceId INTEGER,
            graphNodeId INTEGER, globalTid INTEGER, demangledName TEXT
        )
    """)
    conn.executemany(
        "INSERT INTO CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL VALUES (?,?,?,?,?,?)",
        kernel_rows,
    )
    conn.execute("""
        CREATE TABLE NVTX_EVENTS (
            start INTEGER, end INTEGER, eventType INTEGER,
            globalTid INTEGER, domainId INTEGER, text TEXT
        )
    """)
    conn.commit()
    conn.close()


def _write_annotations(path: str, annotations: dict) -> None:
    with open(path, "w") as f:
        json.dump({str(k): v for k, v in annotations.items()}, f)


class TestInjectNvtxRanges(TestCase):
    def test_adds_nvtx_rows_for_annotated_kernels(self):
        from torch.cuda._annotate_cuda_graph_trace import inject_nvtx_ranges_into_nsys_sqlite

        with tempfile.TemporaryDirectory() as d:
            db = os.path.join(d, "trace.sqlite")
            ann = os.path.join(d, "ann.json")
            out = os.path.join(d, "out.sqlite")

            _make_nsys_db(db, [
                (1000, 2000, 0, 42, 7, "kernel_A"),
                (2100, 3000, 0, 42, 7, "kernel_B"),
                (3500, 4000, 0, 99, 7, "kernel_C"),  # unannotated
            ])
            _write_annotations(ann, {42: {"fqn_map": {"weight": "encoder.weight"}, "graph_id": 0}})

            inject_nvtx_ranges_into_nsys_sqlite(db, ann, out)

            # Original untouched
            orig = sqlite3.connect(db)
            self.assertEqual(orig.execute("SELECT COUNT(*) FROM NVTX_EVENTS").fetchone()[0], 0)
            orig.close()

            result = sqlite3.connect(out)
            rows = result.execute("SELECT start, end, text FROM NVTX_EVENTS").fetchall()
            result.close()

            self.assertGreater(len(rows), 0)
            texts = [r[2] for r in rows]
            self.assertTrue(any("encoder" in (t or "") for t in texts))

    def test_range_spans_all_kernels_in_layer(self):
        from torch.cuda._annotate_cuda_graph_trace import inject_nvtx_ranges_into_nsys_sqlite

        with tempfile.TemporaryDirectory() as d:
            db = os.path.join(d, "trace.sqlite")
            ann = os.path.join(d, "ann.json")
            out = os.path.join(d, "out.sqlite")

            _make_nsys_db(db, [
                (500, 600, 0, 7, 3, "k1"),
                (700, 900, 0, 7, 3, "k2"),
            ])
            _write_annotations(ann, {7: {"fqn_map": {"w": "decoder.weight"}, "graph_id": 1}})

            inject_nvtx_ranges_into_nsys_sqlite(db, ann, out)

            result = sqlite3.connect(out)
            rows = result.execute("SELECT start, end FROM NVTX_EVENTS").fetchall()
            result.close()

            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0][0], 500)  # min kernel start
            self.assertEqual(rows[0][1], 900)  # max kernel end

    def test_no_nvtx_rows_when_no_annotations_match(self):
        from torch.cuda._annotate_cuda_graph_trace import inject_nvtx_ranges_into_nsys_sqlite

        with tempfile.TemporaryDirectory() as d:
            db = os.path.join(d, "trace.sqlite")
            ann = os.path.join(d, "ann.json")
            out = os.path.join(d, "out.sqlite")

            _make_nsys_db(db, [(100, 200, 0, 55, 1, "k")])
            _write_annotations(ann, {99: {"fqn_map": {"w": "other.w"}, "graph_id": 0}})

            inject_nvtx_ranges_into_nsys_sqlite(db, ann, out)

            result = sqlite3.connect(out)
            count = result.execute("SELECT COUNT(*) FROM NVTX_EVENTS").fetchone()[0]
            result.close()
            self.assertEqual(count, 0)

    def test_raises_when_output_equals_input(self):
        from torch.cuda._annotate_cuda_graph_trace import inject_nvtx_ranges_into_nsys_sqlite

        with tempfile.TemporaryDirectory() as d:
            db = os.path.join(d, "trace.sqlite")
            ann = os.path.join(d, "ann.json")
            sqlite3.connect(db).close()
            _write_annotations(ann, {})
            with self.assertRaises(ValueError):
                inject_nvtx_ranges_into_nsys_sqlite(db, ann, db)

    def test_common_prefix_label(self):
        """Labels use the longest common FQN prefix across parameters."""
        from torch.cuda._annotate_cuda_graph_trace import inject_nvtx_ranges_into_nsys_sqlite

        with tempfile.TemporaryDirectory() as d:
            db = os.path.join(d, "trace.sqlite")
            ann = os.path.join(d, "ann.json")
            out = os.path.join(d, "out.sqlite")

            _make_nsys_db(db, [(100, 200, 0, 5, 1, "k")])
            _write_annotations(ann, {
                5: {
                    "fqn_map": {
                        "p1": "encoder.linear.weight",
                        "p2": "encoder.linear.bias",
                    },
                    "graph_id": 0,
                }
            })

            inject_nvtx_ranges_into_nsys_sqlite(db, ann, out)

            result = sqlite3.connect(out)
            texts = [r[0] for r in result.execute("SELECT text FROM NVTX_EVENTS").fetchall()]
            result.close()

            self.assertEqual(len(texts), 1)
            self.assertEqual(texts[0], "encoder.linear")


if __name__ == "__main__":
    run_tests()
