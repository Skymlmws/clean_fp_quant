import pytest

from video_quant_lab.baselines.fp_quant.scripts.generate.generate_wan_vbench_batch import (
    select_records,
)


def test_select_all_preserves_metadata_order() -> None:
    records = [{"source_index": index} for index in range(3)]

    assert select_records(records, 3, 123, "all") == records


def test_select_all_requires_the_complete_dataset() -> None:
    records = [{"source_index": index} for index in range(3)]

    with pytest.raises(ValueError, match="requires prompt-count=3"):
        select_records(records, 2, 123, "all")
