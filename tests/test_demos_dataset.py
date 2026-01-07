from pathlib import Path


def test_demos_dataset_constructs():
    from embodied_ai.data.demos_dataset import DemoDatasetConfig, DemosDataset

    ds = DemosDataset(DemoDatasetConfig(path=Path("data_local/demos/trajectories.npz")))
    assert ds.cfg.path.name.endswith(".npz")


