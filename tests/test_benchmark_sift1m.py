from argparse import Namespace

import numpy as np
import pytest

from scripts.benchmark_sift1m import mmap_vecs, recall_at_k, validate_args


def write_vecs(path, values, dtype):
    values = np.asarray(values, dtype=dtype)
    records = np.empty((len(values), values.shape[1] + 1), dtype=np.int32)
    records[:, 0] = values.shape[1]
    if dtype == np.float32:
        records[:, 1:] = values.view(np.int32)
    else:
        records[:, 1:] = values
    records.tofile(path)


def test_mmap_vecs_reads_fvecs_and_ivecs(tmp_path):
    floats = np.array([[1.5, 2.5], [3.5, 4.5]], dtype=np.float32)
    integers = np.array([[3, 4, 5], [6, 7, 8]], dtype=np.int32)
    write_vecs(tmp_path / "data.fvecs", floats, np.float32)
    write_vecs(tmp_path / "neighbors.ivecs", integers, np.int32)

    loaded_floats, float_width = mmap_vecs(tmp_path / "data.fvecs", np.float32)
    loaded_integers, integer_width = mmap_vecs(tmp_path / "neighbors.ivecs", np.int32)

    assert float_width == 2
    assert integer_width == 3
    np.testing.assert_array_equal(loaded_floats, floats)
    np.testing.assert_array_equal(loaded_integers, integers)


def test_mmap_vecs_rejects_inconsistent_headers(tmp_path):
    path = tmp_path / "bad.fvecs"
    records = np.array([[2, 0, 0], [3, 0, 0]], dtype=np.int32)
    records.tofile(path)

    with pytest.raises(ValueError, match="Inconsistent record headers"):
        mmap_vecs(path, np.float32)


def test_recall_at_k_uses_set_overlap():
    actual = np.array([[1, 2], [8, 9]], dtype=np.int64)
    expected = np.array([[2, 1], [7, 8]], dtype=np.int64)

    assert recall_at_k(actual, expected, 2) == 0.75


def test_validate_args_rejects_k_larger_than_selected_base():
    args = Namespace(
        num_servers=1,
        k=11,
        add_batch_size=10,
        query_batch_size=10,
        max_base=10,
        max_queries=None,
    )

    with pytest.raises(ValueError, match="exceeds the selected base size"):
        validate_args(args, base_count=100, query_count=10, groundtruth_width=100)
