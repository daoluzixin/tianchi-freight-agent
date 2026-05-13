"""供给预测器：基于历史 query_cargo 数据构建「时段×区域」供给质量模型。

核心思想：
  将经纬度按 grid_size 度分桶、将一天 24 小时按 time_slot_hours 分段，
  记录每个 (时段, 网格) 的历史货源数量和平均质量（价格/距离比），
  用于预测"在某个位置等待 N 分钟后能获得多好的货源"。

用途：
  1. 替代 wait_value 的盲猜 → 基于数据的期望值估计
  2. 评估卸货点的未来价值 → 改善 position_bonus
  3. 指导 query 冷却策略 → 高供给时段多查、低供给时段少查
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any


@dataclass
class SupplyStats:
    """单个 (时段, 网格) 桶的供给统计。"""
    total_count: int = 0          # 累计货源数量
    total_quality: float = 0.0    # 累计质量分（price / max(pickup_km, 1)）
    observation_count: int = 0    # 观测次数（query 次数）

    @property
    def avg_quality(self) -> float:
        """平均单条货源质量。"""
        if self.total_count == 0:
            return 0.0
        return self.total_quality / self.total_count

    @property
    def avg_density(self) -> float:
        """每次查询的平均货源密度。"""
        if self.observation_count == 0:
            return 0.0
        return self.total_count / self.observation_count

    def decay(self, factor: float) -> None:
        """时间衰减，让近期数据权重更高。"""
        self.total_count = int(self.total_count * factor)
        self.total_quality *= factor
        self.observation_count = int(self.observation_count * factor)


class SupplyPredictor:
    """基于历史数据的供给预测器。

    维护一个二维直方图：(time_slot, grid_cell) → SupplyStats
    其中 time_slot = hour // time_slot_hours，grid_cell = (lat_bucket, lng_bucket)。
    """

    def __init__(
        self,
        grid_size: float = 0.1,        # 约 11km × 10km 一个格（比 HotspotTracker 更细）
        time_slot_hours: int = 2,       # 每 2 小时一个时段（0-2, 2-4, ..., 22-24）
        decay_factor: float = 0.95,     # 每天衰减
    ) -> None:
        self._grid_size = grid_size
        self._time_slot_hours = time_slot_hours
        self._decay_factor = decay_factor
        self._stats: dict[tuple[int, int, int], SupplyStats] = defaultdict(SupplyStats)
        self._last_decay_day: int = -1

        # 全局统计（用于 fallback）
        self._global_by_slot: dict[int, SupplyStats] = defaultdict(SupplyStats)

    def _time_slot(self, sim_minutes: int) -> int:
        """将仿真分钟转为时段索引。"""
        hour = (sim_minutes % 1440) // 60
        return hour // self._time_slot_hours

    def _grid_cell(self, lat: float, lng: float) -> tuple[int, int]:
        """将经纬度转为网格索引。"""
        return (int(lat / self._grid_size), int(lng / self._grid_size))

    def _maybe_decay(self, current_day: int) -> None:
        """每天执行一次衰减。"""
        if current_day > self._last_decay_day:
            for stats in self._stats.values():
                stats.decay(self._decay_factor)
            for stats in self._global_by_slot.values():
                stats.decay(self._decay_factor)
            self._last_decay_day = current_day

    # =========================================================================
    # 数据输入
    # =========================================================================

    def observe(self, cargos: list[dict[str, Any]], sim_minutes: int,
                observer_lat: float, observer_lng: float) -> None:
        """记录一次 query_cargo 的结果。

        Args:
            cargos: query_cargo 返回的货源列表
            sim_minutes: 当前仿真时刻
            observer_lat/lng: 查询时的司机位置
        """
        current_day = sim_minutes // 1440
        self._maybe_decay(current_day)

        slot = self._time_slot(sim_minutes)
        cell = self._grid_cell(observer_lat, observer_lng)
        key = (slot, cell[0], cell[1])

        stats = self._stats[key]
        global_stats = self._global_by_slot[slot]

        stats.observation_count += 1
        global_stats.observation_count += 1

        for cargo in cargos:
            price = float(cargo.get("price", 0.0))
            pickup_km = float(cargo.get("query_distance_km", 0.0))
            if pickup_km <= 0:
                pickup_lat = float(cargo.get("pickup_lat", 0.0))
                pickup_lng = float(cargo.get("pickup_lng", 0.0))
                from agent.core.state_tracker import haversine_km
                pickup_km = haversine_km(observer_lat, observer_lng, pickup_lat, pickup_lng)

            quality = price / max(pickup_km, 1.0)

            stats.total_count += 1
            stats.total_quality += quality
            global_stats.total_count += 1
            global_stats.total_quality += quality

    # =========================================================================
    # 预测接口
    # =========================================================================

    def predict_supply_quality(self, lat: float, lng: float,
                                sim_minutes: int) -> float:
        """预测在指定位置和时段的期望供给质量。

        返回值：期望的单条货源质量分（price / pickup_km）。
        值越高表示该位置该时段的货源越好。
        """
        slot = self._time_slot(sim_minutes)
        cell = self._grid_cell(lat, lng)
        key = (slot, cell[0], cell[1])

        stats = self._stats.get(key)
        if stats and stats.observation_count >= 2:
            return stats.avg_quality

        # 数据不足时：查看相邻格子
        neighbor_quality = self._neighbor_avg_quality(slot, cell)
        if neighbor_quality > 0:
            return neighbor_quality

        # 再 fallback：该时段的全局平均
        global_stats = self._global_by_slot.get(slot)
        if global_stats and global_stats.observation_count >= 1:
            return global_stats.avg_quality

        # 完全无数据：返回保守默认值
        return 50.0

    def predict_supply_density(self, lat: float, lng: float,
                                sim_minutes: int) -> float:
        """预测在指定位置和时段的期望货源密度（每次查询的平均条数）。"""
        slot = self._time_slot(sim_minutes)
        cell = self._grid_cell(lat, lng)
        key = (slot, cell[0], cell[1])

        stats = self._stats.get(key)
        if stats and stats.observation_count >= 2:
            return stats.avg_density

        global_stats = self._global_by_slot.get(slot)
        if global_stats and global_stats.observation_count >= 1:
            return global_stats.avg_density

        return 30.0  # 默认中等密度

    def predict_wait_value(self, lat: float, lng: float,
                           sim_minutes: int, wait_minutes: int = 30) -> float:
        """预测在指定位置等待 wait_minutes 后的期望最优货源得分。

        综合考虑：当前时段供给质量 × 密度调整 × 时间衰减。
        """
        quality = self.predict_supply_quality(lat, lng, sim_minutes)
        density = self.predict_supply_density(lat, lng, sim_minutes)

        # 密度调整：货源越多，等到好单的概率越高
        density_factor = min(1.5, max(0.3, density / 50.0))

        # 等待时间调整：等越久越可能等到好单，但边际递减
        time_factor = min(1.5, 1.0 + math.log1p(wait_minutes / 30.0) * 0.3)

        # 也看看等待结束后的时段（可能跨时段）
        future_quality = self.predict_supply_quality(
            lat, lng, sim_minutes + wait_minutes)
        blended_quality = quality * 0.6 + future_quality * 0.4

        return blended_quality * density_factor * time_factor

    def predict_location_future_value(self, lat: float, lng: float,
                                       sim_minutes: int,
                                       horizon_minutes: int = 120) -> float:
        """预测某个位置在未来 horizon_minutes 内的综合供给价值。

        用于评估"卸货点的未来价值"——到了那个位置后，
        未来 2 小时内能获得多好的货源。
        """
        if horizon_minutes <= 0:
            return 0.0

        # 在未来时间窗内采样多个时段
        samples = []
        step = max(30, self._time_slot_hours * 30)  # 每半个时段采样一次
        t = sim_minutes
        while t < sim_minutes + horizon_minutes:
            q = self.predict_supply_quality(lat, lng, t)
            d = self.predict_supply_density(lat, lng, t)
            # 综合值 = 质量 × 密度因子
            value = q * min(1.5, max(0.3, d / 50.0))
            samples.append(value)
            t += step

        if not samples:
            return 0.0

        # 取加权平均（近期权重更高）
        total = 0.0
        weight_sum = 0.0
        for i, v in enumerate(samples):
            w = 1.0 / (1.0 + i * 0.3)  # 近期权重高
            total += v * w
            weight_sum += w

        return total / weight_sum if weight_sum > 0 else 0.0

    def is_peak_supply(self, lat: float, lng: float, sim_minutes: int) -> bool:
        """判断当前是否处于供给高峰（用于 query 冷却策略）。"""
        density = self.predict_supply_density(lat, lng, sim_minutes)
        return density > 50.0  # 每次查询平均超过 50 条 = 高峰

    def get_supply_richness(self, lat: float, lng: float,
                            sim_minutes: int) -> float:
        """返回 0~1 的供给丰富度指标，用于动态 query 冷却。

        0 = 极度稀疏，1 = 非常丰富。
        """
        density = self.predict_supply_density(lat, lng, sim_minutes)
        # 用 sigmoid 映射到 0~1
        return 1.0 / (1.0 + math.exp(-0.05 * (density - 40)))

    # =========================================================================
    # 内部方法
    # =========================================================================

    def _neighbor_avg_quality(self, slot: int, cell: tuple[int, int]) -> float:
        """查看相邻 8 个格子的平均质量。"""
        total_quality = 0.0
        total_count = 0
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr == 0 and dc == 0:
                    continue
                key = (slot, cell[0] + dr, cell[1] + dc)
                stats = self._stats.get(key)
                if stats and stats.total_count > 0:
                    total_quality += stats.total_quality
                    total_count += stats.total_count

        if total_count > 0:
            return total_quality / total_count
        return 0.0
