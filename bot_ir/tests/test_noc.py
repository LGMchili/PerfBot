"""NoC ring 数据接入：noc_raw 目录（manifest + 每指标一张矩阵 CSV）。"""

import json

import numpy as np
import pandas as pd
import pytest

from profmw import ProfilingStore
from profmw.noc import DEFAULT_LEVELS, add_noc_raw

MODULES = [f"R{r:02d}_{ch}_CS{s:02d}"
           for r in range(2) for ch in ("Req", "Snp") for s in range(3)]
TIMES = [0.0, 100.0, 200.0]


def make_raw_dir(tmp_path, modules=MODULES, times=TIMES, manifest=None,
                 skip_files=()):
    raw = tmp_path / "noc_raw"
    raw.mkdir()
    types = {
        "noc_up_fail_count": {"unit": "count", "stat": "窗口内失败次数"},
        "noc_left_util": {"unit": "%", "stat": "窗口内平均占用率"},
    }
    if manifest is not None:
        types = manifest
    rng = np.random.default_rng(0)
    for ptype in types:
        if ptype in skip_files:
            continue
        df = pd.DataFrame(rng.uniform(0, 10, (len(modules), len(times))),
                          index=modules, columns=times)
        df.to_csv(raw / f"{ptype}.csv")
    (raw / "manifest.json").write_text(json.dumps(types, ensure_ascii=False),
                                       encoding="utf-8")
    return raw, types


class TestAddNocRaw:
    def test_ingests_all_types_with_meta(self, tmp_path):
        raw, types = make_raw_dir(tmp_path)
        store = ProfilingStore(meta={"benchmark": "noc"})
        add_noc_raw(store, raw)
        assert set(store.list_types()) == set(types)
        assert store.list_types()["noc_left_util"]["unit"] == "%"
        assert store.get_matrix("noc_up_fail_count").shape == (len(MODULES), len(TIMES))

    def test_levels_written_to_run_meta(self, tmp_path):
        raw, _ = make_raw_dir(tmp_path)
        store = ProfilingStore()
        add_noc_raw(store, raw)
        assert store.meta["levels"] == DEFAULT_LEVELS

    def test_row_labels_must_conform_to_levels(self, tmp_path):
        raw, _ = make_raw_dir(tmp_path, modules=["bad_label", "R00_Req_CS00", "x"])
        store = ProfilingStore()
        with pytest.raises(ValueError, match="行标.*不符合"):
            add_noc_raw(store, raw)

    def test_conflicting_existing_levels_rejected(self, tmp_path):
        raw, _ = make_raw_dir(tmp_path)
        store = ProfilingStore(meta={"levels": [{"key": "X", "name": "other",
                                                 "width": 2}]})
        with pytest.raises(ValueError, match="levels"):
            add_noc_raw(store, raw)

    def test_missing_manifest_raises(self, tmp_path):
        raw = tmp_path / "empty"
        raw.mkdir()
        store = ProfilingStore()
        with pytest.raises(FileNotFoundError, match="manifest"):
            add_noc_raw(store, raw)

    def test_manifest_file_missing_csv_raises(self, tmp_path):
        manifest = {"noc_up_fail_count": {"unit": "count", "stat": "x"},
                    "noc_ghost": {"unit": "count", "stat": "x"}}
        raw, _ = make_raw_dir(tmp_path, manifest=manifest,
                              skip_files=("noc_ghost",))
        store = ProfilingStore()
        with pytest.raises(FileNotFoundError, match="noc_ghost"):
            add_noc_raw(store, raw)

    def test_roundtrip_and_prefix_query(self, tmp_path):
        raw, _ = make_raw_dir(tmp_path)
        store = ProfilingStore()
        add_noc_raw(store, raw)
        run = tmp_path / "run_noc"
        store.save(run)
        loaded = ProfilingStore.load(run)
        assert loaded.meta["levels"] == DEFAULT_LEVELS  # levels 随 meta.json 往返
        val = loaded.aggregate("noc_up_fail_count", module_prefix="R01_", agg="sum")
        manual = loaded.get_matrix("noc_up_fail_count")
        rows = [m for m in manual.index if m.startswith("R01_")]
        assert val == pytest.approx(manual.loc[rows].sum().sum())


# ---------------------------------------------------------------- dummy NoC 生成

# 4 个失败计数（In/Out × Up/Down）+ 2 个链路占用率
NOC_TYPES = ["noc_UpInFail", "noc_UpOutFail", "noc_DownInFail", "noc_DownOutFail",
             "noc_left_util", "noc_right_util"]


class TestMakeNocRawDir:
    def test_layout_8rings_4channels_25cs(self, tmp_path):
        from profmw.dummy import make_noc_raw_dir
        raw = make_noc_raw_dir(tmp_path)
        assert (raw / "manifest.json").exists()
        for ptype in NOC_TYPES:
            assert (raw / f"{ptype}.csv").exists()
        df = pd.read_csv(raw / "noc_UpOutFail.csv", index_col=0)
        assert df.shape[0] == 8 * 4 * 25
        assert df.index[0] == "R00_Req_CS00"
        assert df.index[-1] == "R07_Dat_CS24"
        assert {m.split("_")[1] for m in df.index} == {"Req", "Snp", "Rsp", "Dat"}

    def test_manifest_declares_meta(self, tmp_path):
        from profmw.dummy import make_noc_raw_dir
        raw = make_noc_raw_dir(tmp_path)
        manifest = json.loads((raw / "manifest.json").read_text(encoding="utf-8"))
        assert set(manifest) == set(NOC_TYPES)
        assert manifest["noc_UpOutFail"]["unit"] == "count"
        assert manifest["noc_left_util"]["unit"] == "%"
        assert manifest["noc_UpInFail"]["module"] == "dummy"  # 模块分组字段

    def test_node_names_json_in_raw_dir(self, tmp_path):
        # CS 槽位的硬件 IP 名（HBMC/MATA/SMMU...），按 ring+CS 定位
        from profmw.dummy import make_noc_raw_dir
        raw = make_noc_raw_dir(tmp_path)
        f = raw / "node_names.json"
        assert f.exists()
        names = json.loads(f.read_text(encoding="utf-8"))
        assert len(names) == 8 * 25
        assert "R00_CS00" in names and "R07_CS24" in names   # 键不带通道
        assert {"HBMC", "MATA", "SMMU"} <= set(names.values())

    def test_add_noc_raw_loads_node_names(self, tmp_path):
        from profmw.dummy import make_noc_raw_dir
        raw = make_noc_raw_dir(tmp_path)
        store = ProfilingStore()
        add_noc_raw(store, raw)
        names = store.meta["noc_node_names"]
        assert names["R03_CS13"]                     # 拥塞注入点也有 IP 名
        run = tmp_path / "run"
        store.save(run)
        assert ProfilingStore.load(run).meta["noc_node_names"] == names  # 往返保留

    def test_ring_die_json_in_raw_dir(self, tmp_path):
        # ring 所属 die：2 die × 4 ring，键为 ring 名
        from profmw.dummy import make_noc_raw_dir
        raw = make_noc_raw_dir(tmp_path)
        f = raw / "ring_die.json"
        assert f.exists()
        dies = json.loads(f.read_text(encoding="utf-8"))
        assert set(dies) == {f"R{r:02d}" for r in range(8)}
        assert dies["R00"] == "die0" and dies["R04"] == "die1"
        assert len(set(dies.values())) == 2

    def test_add_noc_raw_loads_ring_die(self, tmp_path):
        from profmw.dummy import make_noc_raw_dir
        raw = make_noc_raw_dir(tmp_path)
        store = ProfilingStore()
        add_noc_raw(store, raw)
        dies = store.meta["noc_ring_die"]
        assert dies["R07"] == "die1"
        run = tmp_path / "run"
        store.save(run)
        assert ProfilingStore.load(run).meta["noc_ring_die"] == dies  # 往返保留

    def test_anomaly_congested_segment_stands_out(self, tmp_path):
        from profmw.dummy import make_noc_raw_dir
        raw = make_noc_raw_dir(tmp_path, anomaly=True)
        df = pd.read_csv(raw / "noc_UpOutFail.csv", index_col=0)
        row_means = df.mean(axis=1)
        hot = row_means["R03_Rsp_CS13"]      # 注入拥塞的环段中点
        median = row_means.median()
        assert hot > 10 * median
        down_in = pd.read_csv(raw / "noc_DownInFail.csv", index_col=0)
        assert down_in.mean(axis=1)["R03_Rsp_CS13"] > 10 * down_in.mean(axis=1).median()
        util = pd.read_csv(raw / "noc_left_util.csv", index_col=0)
        assert util.mean(axis=1)["R03_Rsp_CS13"] > 80

    def test_healthy_without_anomaly(self, tmp_path):
        from profmw.dummy import make_noc_raw_dir
        raw = make_noc_raw_dir(tmp_path, anomaly=False)
        df = pd.read_csv(raw / "noc_UpOutFail.csv", index_col=0)
        row_means = df.mean(axis=1)
        assert row_means.max() < 10 * row_means.median() + 1

    def test_deterministic(self, tmp_path):
        from profmw.dummy import make_noc_raw_dir
        a = pd.read_csv(make_noc_raw_dir(tmp_path / "a", seed=3)
                        / "noc_left_util.csv", index_col=0)
        b = pd.read_csv(make_noc_raw_dir(tmp_path / "b", seed=3)
                        / "noc_left_util.csv", index_col=0)
        pd.testing.assert_frame_equal(a, b)

    def test_ingestible_via_add_noc_raw(self, tmp_path):
        from profmw.dummy import make_noc_raw_dir
        raw = make_noc_raw_dir(tmp_path)
        store = ProfilingStore(meta={"benchmark": "noc_dummy"})
        add_noc_raw(store, raw)
        assert set(store.list_types()) == set(NOC_TYPES)
        assert store.meta["levels"][0]["key"] == "R"
        assert store.get_matrix("noc_right_util").shape == (800, 200)
