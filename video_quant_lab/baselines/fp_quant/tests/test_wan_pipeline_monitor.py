from scripts.monitor_wan_baseline_pipeline import schedulable_count


def test_pipeline_leaves_two_gpus_when_machine_is_idle():
    assert schedulable_count(pool_count=8, idle_count=8, shared_gpu_target=2) == 6


def test_pipeline_leaves_one_more_gpu_when_one_is_occupied():
    assert schedulable_count(pool_count=8, idle_count=7, shared_gpu_target=2) == 6


def test_pipeline_can_use_all_idle_gpus_when_two_are_occupied():
    assert schedulable_count(pool_count=8, idle_count=6, shared_gpu_target=2) == 6


def test_pipeline_can_use_all_idle_gpus_when_more_than_two_are_occupied():
    assert schedulable_count(pool_count=8, idle_count=4, shared_gpu_target=2) == 4
