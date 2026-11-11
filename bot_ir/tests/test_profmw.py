"""profmw 测试：存储、raw 接入、查询聚合、插值、校验、dummy 数据。"""

import numpy as np
import pandas as pd
import pytest

from profmw import ProfilingStore, ValidationReport


# ---------------------------------------------------------------- helpers

def make_small_matrix():
    """2 模块 × 3 时间点的小矩阵，值便于手算。"""
    return pd.DataFrame(
        [[1.0, 2.0, 3.0], [10.0, 20.0, 30.0]],
        index=["CH0", "CH1"],
        columns=[100.0, 200.0, 300.0],
    )


# ---------------------------------------------------------------- add_matrix / get_matrix

class TestAddMatrix:
    def test_get_matrix_returns_same_data_with_float_columns(self):
        store = ProfilingStore()
        store.add_matrix("dram_bw", make_small_matrix(), meta={"unit": "GB/s"})
        out = store.get_matrix("dram_bw")
        assert out.shape == (2, 3)
        assert list(out.columns) == [100.0, 200.0, 300.0]
        assert list(out.index) == ["CH0", "CH1"]
        assert out.loc["CH0", 200.0] == 2.0

    def test_columns_are_sorted_ascending_on_add(self):
        df = make_small_matrix()[[300.0, 100.0, 200.0]]
        store = ProfilingStore()
        store.add_matrix("t", df)
        assert list(store.get_matrix("t").columns) == [100.0, 200.0, 300.0]

    def test_duplicate_module_names_rejected(self):
        df = pd.DataFrame([[1.0, 2.0], [3.0, 4.0]], index=["CH0", "CH0"], columns=[1.0, 2.0])
        store = ProfilingStore()
        with pytest.raises(ValueError, match="行标.*存在重复"):
            store.add_matrix("t", df)

    def test_duplicate_timestamps_rejected(self):
        df = pd.DataFrame([[1.0, 2.0]], index=["CH0"], columns=[1.0, 1.0])
        store = ProfilingStore()
        with pytest.raises(ValueError, match="列标.*存在重复"):
            store.add_matrix("t", df)

    def test_integer_timestamps_normalized_to_float(self):
        # np.int64 列标应归一为 float64，保证与 save/load 往返后的 dtype 一致
        df = pd.DataFrame([[1.0, 2.0]], index=["CH0"],
                          columns=np.array([10, 20], dtype=np.int64))
        store = ProfilingStore()
        store.add_matrix("t", df)
        out = store.get_matrix("t")
        assert out.columns.dtype == np.dtype("float64")
        assert list(out.columns) == [10.0, 20.0]

    def test_integer_time_column_in_records_normalized(self):
        rec = pd.DataFrame({"time": np.array([10, 20], dtype=np.int64),
                            "module": ["CH0", "CH0"], "value": [1.0, 2.0]})
        store = ProfilingStore()
        store.add_type_from_records("t", rec, window=None)
        assert store.get_matrix("t").columns.dtype == np.dtype("float64")

    def test_get_matrix_returns_copy(self):
        store = ProfilingStore()
        store.add_matrix("t", make_small_matrix())
        out = store.get_matrix("t")
        out.iloc[0, 0] = 999.0
        assert store.get_matrix("t").iloc[0, 0] == 1.0

    def test_unknown_type_raises(self):
        store = ProfilingStore()
        with pytest.raises(KeyError):
            store.get_matrix("nope")


# ---------------------------------------------------------------- save / load

class TestSaveLoad:
    def test_roundtrip_preserves_values_meta_and_float_columns(self, tmp_path):
        store = ProfilingStore(meta={"benchmark": "bw_gather", "chip": "X9-v0.9"})
        store.add_matrix("dram_bw", make_small_matrix(),
                         meta={"unit": "GB/s", "stat": "窗口内计数总和"})
        path = tmp_path / "run_x"
        store.save(path)

        loaded = ProfilingStore.load(path)
        assert loaded.meta == {"benchmark": "bw_gather", "chip": "X9-v0.9"}
        assert loaded.list_types() == {"dram_bw": {"unit": "GB/s", "stat": "窗口内计数总和"}}
        out = loaded.get_matrix("dram_bw")
        pd.testing.assert_frame_equal(out, make_small_matrix())

    def test_roundtrip_preserves_float_precision(self, tmp_path):
        stamps = [0.1 + i * 0.7 for i in range(5)]  # 非整、有精度的时间戳
        df = pd.DataFrame(np.arange(10.0).reshape(2, 5), index=["A", "B"], columns=stamps)
        store = ProfilingStore()
        store.add_matrix("t", df)
        path = tmp_path / "run_p"
        store.save(path)
        loaded = ProfilingStore.load(path)
        assert list(loaded.get_matrix("t").columns) == stamps

    def test_zip_pack_roundtrip(self, tmp_path):
        store = ProfilingStore(meta={"run": 1})
        store.add_matrix("t", make_small_matrix())
        store.save(tmp_path / "run_z", zip_pack=True)
        assert (tmp_path / "run_z.zip").exists()
        assert not (tmp_path / "run_z").exists()
        loaded = ProfilingStore.load(tmp_path / "run_z.zip")
        pd.testing.assert_frame_equal(loaded.get_matrix("t"), make_small_matrix())

    def test_zip_guard_checks_actual_zip_name(self, tmp_path):
        # 目标名为 run.v2 时，守卫必须检查 run.v2.zip 而不是 with_suffix 替换出的 run.zip
        store = ProfilingStore()
        store.add_matrix("t", make_small_matrix())
        store.save(tmp_path / "run.v2", zip_pack=True)
        assert (tmp_path / "run.v2.zip").exists()
        store2 = ProfilingStore()
        store2.add_matrix("t", make_small_matrix())
        with pytest.raises(FileExistsError):
            store2.save(tmp_path / "run.v2", zip_pack=True)

    def test_load_missing_path_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            ProfilingStore.load(tmp_path / "nope")

    def test_on_disk_format_is_csv_with_meta_json(self, tmp_path):
        store = ProfilingStore(meta={"benchmark": "b"})
        store.add_matrix("dram_bw", make_small_matrix())
        path = tmp_path / "run_csv"
        store.save(path)
        assert (path / "meta.json").exists()
        assert (path / "types" / "dram_bw.csv").exists()
        text = (path / "types" / "dram_bw.csv").read_text(encoding="utf-8")
        assert text.splitlines()[0].startswith("module,")  # 行标列名 + float 时间戳表头
        assert "CH0" in text

    def test_handmade_csv_dir_loads(self, tmp_path):
        # 模拟手工准备的 raw CSV + meta.json，无需经过 save 即可 load
        d = tmp_path / "handmade"
        (d / "types").mkdir(parents=True)
        make_small_matrix().to_csv(d / "types" / "t.csv")
        (d / "meta.json").write_text(
            '{"run": {"benchmark": "x"}, "types": {"t": {"unit": "GB/s"}}}',
            encoding="utf-8",
        )
        loaded = ProfilingStore.load(d)
        pd.testing.assert_frame_equal(loaded.get_matrix("t"), make_small_matrix())
        assert loaded.list_types()["t"] == {"unit": "GB/s"}


# ---------------------------------------------------------------- list_types

class TestListTypes:
    def test_lists_all_types_with_meta(self):
        store = ProfilingStore()
        store.add_matrix("a", make_small_matrix(), meta={"unit": "%"})
        store.add_matrix("b", make_small_matrix())
        assert set(store.list_types()) == {"a", "b"}
        assert store.list_types()["a"] == {"unit": "%"}


# ---------------------------------------------------------------- raw 长表接入

def make_small_records():
    """4 条事件：CH0 两条落在窗口 0，一条落在窗口 100；CH1 一条落在窗口 100。"""
    return pd.DataFrame(
        {
            "time": [10.0, 50.0, 150.0, 199.0],
            "module": ["CH0", "CH0", "CH0", "CH1"],
            "value": [1.0, 2.0, 5.0, 7.0],
        }
    )


class TestAddTypeFromRecords:
    def test_bucket_by_window_sum(self):
        store = ProfilingStore()
        store.add_type_from_records("dram_bw", make_small_records(), window=100.0)
        df = store.get_matrix("dram_bw")
        assert list(df.columns) == [0.0, 100.0]  # 列标 = 窗口起点
        assert df.loc["CH0", 0.0] == 3.0
        assert df.loc["CH0", 100.0] == 5.0
        assert np.isnan(df.loc["CH1", 0.0])
        assert df.loc["CH1", 100.0] == 7.0

    def test_bucket_by_window_mean(self):
        store = ProfilingStore()
        store.add_type_from_records("t", make_small_records(), window=100.0, bucket_how="mean")
        df = store.get_matrix("t")
        assert df.loc["CH0", 0.0] == 1.5
        assert df.loc["CH0", 100.0] == 5.0

    def test_window_none_pivots_on_record_times(self):
        store = ProfilingStore()
        store.add_type_from_records("t", make_small_records(), window=None)
        df = store.get_matrix("t")
        assert list(df.columns) == [10.0, 50.0, 150.0, 199.0]
        assert df.loc["CH0", 10.0] == 1.0
        assert df.loc["CH1", 199.0] == 7.0

    def test_custom_column_names(self):
        store = ProfilingStore()
        rec = make_small_records().rename(
            columns={"time": "ts", "module": "mod", "value": "v"}
        )
        store.add_type_from_records("t", rec, window=100.0,
                                    time_col="ts", module_col="mod", value_col="v")
        assert store.get_matrix("t").loc["CH0", 0.0] == 3.0

    def test_meta_recorded(self):
        store = ProfilingStore()
        store.add_type_from_records("t", make_small_records(), window=100.0,
                                    meta={"unit": "GB/s"})
        assert store.list_types()["t"] == {"unit": "GB/s"}

    def test_invalid_bucket_how_rejected(self):
        store = ProfilingStore()
        with pytest.raises(ValueError, match="bucket_how"):
            store.add_type_from_records("t", make_small_records(), window=100.0,
                                        bucket_how="median")


# ---------------------------------------------------------------- 聚合查询

def make_store_with_nan():
    """CH0: [1, 2, 3]; CH1: [NaN, 20, 30]，时间点 100/200/300。"""
    df = pd.DataFrame(
        [[1.0, 2.0, 3.0], [np.nan, 20.0, 30.0]],
        index=["CH0", "CH1"],
        columns=[100.0, 200.0, 300.0],
    )
    store = ProfilingStore()
    store.add_matrix("t", df)
    return store


class TestAggregate:
    def test_sum_over_all_modules_and_time(self):
        store = ProfilingStore()
        store.add_matrix("t", make_small_matrix())  # CH0:1,2,3 CH1:10,20,30
        assert store.aggregate("t", agg="sum") == 66.0

    def test_sum_with_time_range(self):
        store = ProfilingStore()
        store.add_matrix("t", make_small_matrix())
        assert store.aggregate("t", t_range=(100.0, 200.0), agg="sum") == 33.0

    def test_sum_with_explicit_modules(self):
        store = ProfilingStore()
        store.add_matrix("t", make_small_matrix())
        assert store.aggregate("t", modules=["CH0"], agg="sum") == 6.0

    def test_module_prefix_filter(self):
        store = ProfilingStore()
        store.add_matrix("t", make_small_matrix())
        assert store.aggregate("t", module_prefix="CH1", agg="sum") == 60.0

    def test_modules_and_prefix_intersect(self):
        store = ProfilingStore()
        store.add_matrix("t", make_small_matrix())
        val = store.aggregate("t", modules=["CH0", "CH1"], module_prefix="CH0", agg="sum")
        assert val == 6.0

    def test_var_matches_pandas_sample_variance(self):
        store = ProfilingStore()
        store.add_matrix("t", make_small_matrix())
        expected = pd.Series([1, 2, 3, 10, 20, 30], dtype=float).var()
        assert store.aggregate("t", agg="var") == expected

    def test_max_min_count(self):
        store = ProfilingStore()
        store.add_matrix("t", make_small_matrix())
        assert store.aggregate("t", agg="max") == 30.0
        assert store.aggregate("t", agg="min") == 1.0
        assert store.aggregate("t", agg="count") == 6

    def test_skipna_true_ignores_nan(self):
        store = make_store_with_nan()
        assert store.aggregate("t", agg="sum") == 56.0  # 1+2+3+20+30
        assert store.aggregate("t", agg="count") == 5

    def test_skipna_false_propagates_nan(self):
        store = make_store_with_nan()
        assert np.isnan(store.aggregate("t", agg="sum", skipna=False))
        assert np.isnan(store.aggregate("t", agg="min", skipna=False))

    def test_open_ended_time_range(self):
        store = ProfilingStore()
        store.add_matrix("t", make_small_matrix())
        assert store.aggregate("t", t_range=(150.0, None), agg="sum") == 55.0
        assert store.aggregate("t", t_range=(None, 150.0), agg="sum") == 11.0

    def test_unknown_agg_rejected(self):
        store = ProfilingStore()
        store.add_matrix("t", make_small_matrix())
        with pytest.raises(ValueError, match="agg"):
            store.aggregate("t", agg="p50")

    def test_unknown_type_raises(self):
        store = ProfilingStore()
        with pytest.raises(KeyError):
            store.aggregate("nope")


class TestWindowSeries:
    def test_full_series_without_filters(self):
        store = ProfilingStore()
        store.add_matrix("t", make_small_matrix())
        pd.testing.assert_frame_equal(store.window_series("t"), make_small_matrix())

    def test_slice_by_modules_and_time_range(self):
        store = ProfilingStore()
        store.add_matrix("t", make_small_matrix())
        out = store.window_series("t", modules=["CH1"], t_range=(150.0, 250.0))
        expected = pd.DataFrame([[20.0]], index=["CH1"], columns=[200.0])
        pd.testing.assert_frame_equal(out, expected)

    def test_returns_copy(self):
        store = ProfilingStore()
        store.add_matrix("t", make_small_matrix())
        out = store.window_series("t")
        out.iloc[0, 0] = 999.0
        assert store.get_matrix("t").iloc[0, 0] == 1.0


# ---------------------------------------------------------------- 边界插值

def make_interp_store():
    """CH0: 1,2,3,4；CH1: 10,20,30,40；时间点 0/100/200/300。"""
    df = pd.DataFrame(
        [[1.0, 2.0, 3.0, 4.0], [10.0, 20.0, 30.0, 40.0]],
        index=["CH0", "CH1"],
        columns=[0.0, 100.0, 200.0, 300.0],
    )
    store = ProfilingStore()
    store.add_matrix("t", df)
    return store


class TestInterp:
    def test_linear_interp_adds_both_boundary_points(self):
        store = make_interp_store()
        out = store.window_series("t", t_range=(50.0, 250.0), interp="linear")
        assert list(out.columns) == [50.0, 100.0, 200.0, 250.0]
        assert list(out.loc["CH0"]) == [1.5, 2.0, 3.0, 3.5]
        assert list(out.loc["CH1"]) == [15.0, 20.0, 30.0, 35.0]

    def test_boundary_on_exact_timestamp_adds_no_virtual_point(self):
        store = make_interp_store()
        out = store.window_series("t", t_range=(100.0, 250.0), interp="linear")
        assert list(out.columns) == [100.0, 200.0, 250.0]
        assert list(out.loc["CH0"]) == [2.0, 3.0, 3.5]

    def test_boundary_outside_data_range_not_extrapolated(self):
        store = make_interp_store()
        out = store.window_series("t", t_range=(-50.0, 250.0), interp="linear")
        assert list(out.columns) == [0.0, 100.0, 200.0, 250.0]

    def test_range_inside_single_interval_yields_two_virtual_points(self):
        store = make_interp_store()
        out = store.window_series("t", t_range=(50.0, 60.0), interp="linear")
        assert list(out.columns) == [50.0, 60.0]
        assert list(out.loc["CH0"]) == [1.5, 1.6]

    def test_nan_neighbor_gives_nan_boundary(self):
        df = pd.DataFrame(
            [[1.0, 2.0, 3.0, 4.0], [np.nan, 20.0, 30.0, 40.0]],
            index=["CH0", "CH1"],
            columns=[0.0, 100.0, 200.0, 300.0],
        )
        store = ProfilingStore()
        store.add_matrix("t", df)
        out = store.window_series("t", t_range=(50.0, 250.0), interp="linear")
        assert np.isnan(out.loc["CH1", 50.0])  # 左邻 0 处为 NaN → 边界不插值

    def test_aggregate_with_interp(self):
        store = make_interp_store()
        with_interp = store.aggregate("t", t_range=(50.0, 250.0), agg="sum", interp="linear")
        assert with_interp == 110.0  # (1.5+2+3+3.5)+(15+20+30+35)
        without = store.aggregate("t", t_range=(50.0, 250.0), agg="sum")
        assert without == 55.0

    def test_unknown_interp_rejected(self):
        store = make_interp_store()
        with pytest.raises(ValueError, match="interp"):
            store.window_series("t", t_range=(50.0, 250.0), interp="cubic")

    def test_degenerate_range_t0_equals_t1_yields_single_virtual_point(self):
        store = make_interp_store()
        out = store.window_series("t", t_range=(50.0, 50.0), interp="linear")
        assert list(out.columns) == [50.0]
        assert out.loc["CH0", 50.0] == 1.5
        # aggregate 不应重复计数：CH0@50=1.5 + CH1@50=15.0
        assert store.aggregate("t", t_range=(50.0, 50.0), agg="sum", interp="linear") == 16.5

    def test_inverted_time_range_rejected(self):
        store = make_interp_store()
        with pytest.raises(ValueError, match="t_range"):
            store.window_series("t", t_range=(250.0, 50.0))

    def test_time_range_outside_data_raises_clear_error(self):
        store = make_interp_store()  # 数据时间范围 0~300
        with pytest.raises(ValueError, match="未覆盖"):
            store.aggregate("t", t_range=(500.0, 900.0), agg="max")


# ---------------------------------------------------------------- 校验

class TestValidate:
    def test_clean_store_passes(self):
        store = ProfilingStore()
        store.add_matrix("t", make_small_matrix())
        report = store.validate()
        assert isinstance(report, ValidationReport)
        assert report.ok
        assert len(report.issues) == 1 and report.issues[0].level == "ok"

    def test_high_nan_ratio_warns(self):
        df = pd.DataFrame(
            [[1.0, np.nan], [np.nan, np.nan]],  # NaN 占比 0.75 > 0.5
            index=["A", "B"],
            columns=[1.0, 2.0],
        )
        store = ProfilingStore()
        store.add_matrix("t", df)
        report = store.validate()
        assert not report.ok
        assert any(i.level == "warn" and "NaN" in i.message for i in report.issues)

    def test_inf_is_error(self):
        df = pd.DataFrame([[np.inf, 2.0]], index=["A"], columns=[1.0, 2.0])
        store = ProfilingStore()
        store.add_matrix("t", df)
        report = store.validate()
        assert any(i.level == "error" for i in report.issues)

    def test_empty_matrix_is_error(self):
        store = ProfilingStore()
        store.add_matrix("t", pd.DataFrame(index=[], columns=[], dtype=float))
        report = store.validate()
        assert any(i.level == "error" for i in report.issues)

    def test_report_str_contains_issues(self):
        df = pd.DataFrame([[np.inf, 2.0]], index=["A"], columns=[1.0, 2.0])
        store = ProfilingStore()
        store.add_matrix("t", df)
        text = str(store.validate())
        assert "ERROR" in text and "t" in text


# ---------------------------------------------------------------- dummy 数据

class TestDummy:
    def test_make_dummy_store_shapes(self):
        from profmw.dummy import make_dummy_store
        store = make_dummy_store()
        types = store.list_types()
        assert set(types) == {"ddr_bw", "core_read_outstanding"}
        ddr = store.get_matrix("ddr_bw")
        cores = store.get_matrix("core_read_outstanding")
        assert ddr.shape[0] == 96 and ddr.index[0] == "CH00"
        assert cores.shape[0] == 120 and cores.index[0] == "CORE000"
        for df in (ddr, cores):
            assert list(df.columns) == sorted(df.columns)  # 时间轴升序

    def test_dummy_meta_present(self):
        from profmw.dummy import make_dummy_store
        store = make_dummy_store()
        assert store.meta["benchmark"] == "dummy"
        assert store.list_types()["ddr_bw"]["unit"] == "GB/s"
        assert store.list_types()["core_read_outstanding"]["unit"] == "count"

    def test_default_channels_are_balanced(self):
        from profmw.dummy import make_dummy_store
        store = make_dummy_store()
        ch_means = store.get_matrix("ddr_bw").mean(axis=1)
        spread = ch_means.max() / ch_means.min()
        assert spread < 3.0  # 均匀分布下通道间不应有极端失衡

    def test_aliasing_mode_creates_extreme_channel_imbalance(self):
        from profmw.dummy import make_dummy_store
        normal = make_dummy_store(aliasing=False).get_matrix("ddr_bw").mean(axis=1)
        aliased = make_dummy_store(aliasing=True).get_matrix("ddr_bw").mean(axis=1)
        assert aliased.max() / aliased.median() > 20.0  # 混叠：热点通道远超中位
        var_normal = normal.var()
        var_aliased = aliased.var()
        assert var_aliased > 10 * var_normal

    def test_hotspot_mode_creates_hot_cores(self):
        from profmw.dummy import make_dummy_store
        core_means = make_dummy_store(hotspot=True).get_matrix(
            "core_read_outstanding").mean(axis=1)
        assert core_means.max() / core_means.median() > 5.0

    def test_make_dummy_records_is_ingestible(self):
        from profmw.dummy import make_dummy_records
        rec = make_dummy_records("ddr_bw")
        assert set(["time", "module", "value"]) <= set(rec.columns)
        store = ProfilingStore()
        store.add_type_from_records("ddr_bw", rec, window=100.0, bucket_how="mean",
                                    meta={"unit": "GB/s"})
        df = store.get_matrix("ddr_bw")
        assert df.shape[0] == 96
        assert df.shape[1] > 10  # 多个窗口

    def test_dummy_is_deterministic(self):
        from profmw.dummy import make_dummy_store
        a = make_dummy_store(seed=7).get_matrix("ddr_bw")
        b = make_dummy_store(seed=7).get_matrix("ddr_bw")
        pd.testing.assert_frame_equal(a, b)
        c = make_dummy_store(seed=8).get_matrix("ddr_bw")
        assert not a.equals(c)
