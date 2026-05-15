"""货源评分器：对通过 RuleEngine 的候选货源计算综合得分并排序。

得分公式基于：
  score = gross_income - total_cost - penalty_risk + position_bonus

其中考虑：运费收入、空驶成本、时间成本、位置优势（卸货后靠近货源密集区）、
以及偏好违反的罚分风险。

所有数值权重均由 StrategyParams 驱动，CargoScorer 自身不含硬编码常量。
位置加分通过 HotspotTracker 从 query_cargo 历史中动态学习，不依赖固定坐标。
"""

from __future__ import annotations

import math
from collections import defaultdict
from typing import TYPE_CHECKING, Any

from agent.config.driver_config import DriverConfig, _parse_datetime_to_sim_minutes
from agent.core.rule_engine import FilteredCargo
from agent.core.state_tracker import DriverState, haversine_km
from agent.core.timeline_projector import (
    SCAN_COST_MINUTES,
    compute_go_home_penalty_score,
)
from agent.scoring.supply_predictor import SupplyPredictor

if TYPE_CHECKING:
    from agent.scoring.experience_tracker import ExperienceSummary, DeliveryRegionSummary
    from agent.strategy.strategy_advisor import StrategyParams


class ScoredCargo:
    """评分后的候选货源。"""

    def __init__(self, filtered: FilteredCargo, score: float, breakdown: dict[str, float]):
        self.filtered = filtered
        self.cargo = filtered.cargo
        self.score = score
        self.breakdown = breakdown  # 各维度得分明细
        self.pickup_km = filtered.pickup_km
        self.haul_km = filtered.haul_km

    def __repr__(self) -> str:
        cargo_id = self.cargo.get("cargo_id", "?")
        return f"ScoredCargo({cargo_id}, score={self.score:.2f})"


# ---------------------------------------------------------------------------
# 动态货源热点追踪器
# ---------------------------------------------------------------------------

class HotspotTracker:
    """从 query_cargo 历史中学习货源密集区，替代硬编码坐标。

    原理：将经纬度按 grid_size 度分桶，统计每个桶的货源出现次数，
    取 top-N 作为热点中心。桶的中心坐标即为热点坐标。
    """

    def __init__(self, grid_size: float = 0.15, top_n: int = 5, decay: float = 0.98) -> None:
        self._grid_size = grid_size
        self._top_n = top_n
        self._decay = decay  # 每天衰减因子，让近期数据权重更高
        self._buckets: dict[tuple[int, int], float] = defaultdict(float)
        self._hotspots: list[tuple[float, float]] = []
        self._last_decay_day: int = -1

    def observe(self, cargos: list[dict[str, Any]], current_day: int) -> None:
        """记录一批货源的位置信息。"""
        # 每天执行一次衰减
        if current_day > self._last_decay_day:
            for key in self._buckets:
                self._buckets[key] *= self._decay
            self._last_decay_day = current_day

        for cargo in cargos:
            lat = float(cargo.get("pickup_lat", 0.0))
            lng = float(cargo.get("pickup_lng", 0.0))
            if lat == 0.0 and lng == 0.0:
                continue
            bucket = (int(lat / self._grid_size), int(lng / self._grid_size))
            self._buckets[bucket] += 1.0

        self._refresh_hotspots()

    def get_hotspots(self) -> list[tuple[float, float]]:
        """返回当前识别到的热点中心列表。"""
        return self._hotspots

    def _refresh_hotspots(self) -> None:
        if not self._buckets:
            return
        sorted_buckets = sorted(self._buckets.items(), key=lambda x: x[1], reverse=True)
        self._hotspots = []
        for (r, c), _count in sorted_buckets[:self._top_n]:
            center_lat = (r + 0.5) * self._grid_size
            center_lng = (c + 0.5) * self._grid_size
            self._hotspots.append((center_lat, center_lng))


# ---------------------------------------------------------------------------
# CargoScorer
# ---------------------------------------------------------------------------

class CargoScorer:
    """多维度货源评分器。

    所有权重/阈值通过 StrategyParams 注入，不含硬编码常量。
    位置加分通过 HotspotTracker 动态学习。
    """

    def __init__(self, cost_per_km: float = 1.5) -> None:
        self.cost_per_km = cost_per_km
        self.hotspot_tracker = HotspotTracker()
        self.supply_predictor = SupplyPredictor()
        # 由 ModelDecisionService 在每步决策前注入当前场景的经验摘要
        self._current_experience: "ExperienceSummary | None" = None

    def score_and_rank(self, candidates: list[FilteredCargo],
                       state: DriverState, config: DriverConfig,
                       top_k: int = 5,
                       params: "StrategyParams | None" = None) -> list[ScoredCargo]:
        """对候选货源评分并返回 top-k。"""
        scored: list[ScoredCargo] = []

        for fc in candidates:
            score, breakdown = self._compute_score(fc, state, config, params)
            scored.append(ScoredCargo(filtered=fc, score=score, breakdown=breakdown))

        # 按 score 降序排列
        scored.sort(key=lambda x: x.score, reverse=True)
        return scored[:top_k]

    # ------------------------------------------------------------------
    # 核心评分
    # ------------------------------------------------------------------

    def _compute_score(self, fc: FilteredCargo, state: DriverState,
                       config: DriverConfig,
                       params: "StrategyParams | None" = None) -> tuple[float, dict[str, float]]:
        """计算单条货源的综合得分。

        所有阈值/权重均从 params 读取；params 为 None 时使用默认值。

        """
        # 延迟导入避免循环；若 params 为 None 则构造默认值
        if params is None:
            from agent.strategy.strategy_advisor import StrategyParams
            params = StrategyParams()

        cargo = fc.cargo


        # 1) 运费收入
        gross_income = float(cargo.get("price", 0.0))

        # 2) 总距离成本
        total_km = fc.pickup_km + fc.haul_km
        distance_cost = total_km * self.cost_per_km

        # 3) 时间机会成本（动态校准）
        trip_minutes = total_km / config.reposition_speed_kmpm
        dynamic_time_cost = self._calibrate_time_cost(
            params.time_cost_per_minute, state, params)
        time_cost = trip_minutes * dynamic_time_cost

        # 4) 空驶比惩罚
        deadhead_ratio = fc.pickup_km / max(fc.haul_km, 1.0)
        deadhead_penalty = 0.0
        if deadhead_ratio > params.deadhead_ratio_threshold:
            deadhead_penalty = ((deadhead_ratio - params.deadhead_ratio_threshold)
                                * params.deadhead_over_penalty
                                * params.deadhead_penalty_factor)

        # 5) 偏好违反风险（软约束）
        #    P3 fix: 使用实际罚分金额（如有），而非固定默认值
        preference_penalty = 0.0
        if fc.is_soft_violated:
            actual_amount = fc.soft_violation_amount
            base_penalty = actual_amount if actual_amount > 0 else params.soft_violation_penalty
            # 软约束罚分加重：实际罚分 * 2.5 倍作为评分惩罚
            # 因为接一单软约束货不仅有直接罚分，还有"消耗了一个接单机会"的机会成本
            preference_penalty = base_penalty * 2.5

        # 6) 位置优势：卸货点的未来供给价值（数据驱动 + 热点 fallback）
        delivery_lat = float(cargo.get("delivery_lat", 0.0))
        delivery_lng = float(cargo.get("delivery_lng", 0.0))
        # 估算到达卸货点的时刻
        arrival_sim_min = state.sim_minutes + int(trip_minutes)
        position_bonus = self._enhanced_position_bonus(
            delivery_lat, delivery_lng, arrival_sim_min, params)

        # 7) 空驶配额保护（增强版：渐进式罚分）
        deadhead_budget_penalty = 0.0
        if config.max_monthly_deadhead_km is not None:
            remaining_budget = config.max_monthly_deadhead_km - state.total_deadhead_km
            budget_usage_ratio = 1.0 - remaining_budget / max(config.max_monthly_deadhead_km, 1.0)
            # 渐进式罚分：配额使用超过 50% 就开始施加压力
            if budget_usage_ratio > 0.5:
                # 压力因子：50%→0, 75%→1, 100%→2
                pressure = (budget_usage_ratio - 0.5) * 4.0
                deadhead_budget_penalty = (fc.pickup_km
                                           * params.deadhead_budget_penalty_mult
                                           * (1.0 + pressure))
            # 原有的紧急保护仍然保留
            if remaining_budget < fc.pickup_km * params.deadhead_budget_warning_mult:
                deadhead_budget_penalty = max(
                    deadhead_budget_penalty,
                    fc.pickup_km * params.deadhead_budget_penalty_mult * 3.0)

        # 8) 回家便利性
        home_bonus = 0.0
        if config.must_return_home and config.home_pos:
            dist_to_home = haversine_km(delivery_lat, delivery_lng,
                                        config.home_pos[0], config.home_pos[1])
            if dist_to_home < params.home_bonus_radius_km:
                home_bonus = ((params.home_bonus_radius_km - dist_to_home)
                              * params.home_bonus_per_km)

        # 9) 每公里利润率加成
        profit_per_km = (gross_income - distance_cost) / max(total_km, 1.0)
        efficiency_bonus = (max(0, profit_per_km)
                            * params.profit_per_km_bonus_mult
                            * params.efficiency_bonus_factor)

        # 10) 装货等待时间成本（load_time 感知）
        load_wait_minutes = self._estimate_load_wait(cargo, state, fc.pickup_km, config)
        load_wait_cost = load_wait_minutes * params.time_cost_per_minute
        # 如果今日休息未满足，装货等待可折抵休息需求，减轻惩罚
        rest_deficit = max(0, config.min_continuous_rest_minutes - state.longest_rest_today)
        if rest_deficit > 0 and load_wait_minutes > 0:
            rest_credit = min(load_wait_minutes, rest_deficit)
            # 折抵部分的时间成本减半（等待≈休息，一举两得）
            load_wait_cost -= rest_credit * params.time_cost_per_minute * 0.5
            load_wait_cost = max(0.0, load_wait_cost)

        # 11) 首单 deadline 紧迫加分
        #     当天尚未接到首单且距 deadline 临近时，降低接单门槛
        first_order_urgency = 0.0
        if (config.first_order_deadline_hour is not None
                and state.today_first_order_minute is None):
            hour = state.hour_in_day()
            hours_to_deadline = config.first_order_deadline_hour - hour
            first_order_penalty = config.penalty_weights.get("first_order", 0.0)
            if hours_to_deadline < 0:
                # 已过 deadline：加大激励，罚分已经产生，必须止损
                first_order_urgency = first_order_penalty * 0.5
            elif hours_to_deadline <= 2:
                # 距 deadline 不到 2 小时：按紧迫程度线性加分
                urgency_ratio = 1.0 - hours_to_deadline / 2.0
                first_order_urgency = first_order_penalty * 0.3 * urgency_ratio

        # 12) 休息余量前瞻罚分
        rest_lookahead_penalty = 0.0
        if config.min_continuous_rest_minutes > 0:
            # 估算运输完成时间（含 scan_cost）
            trip_minutes_est = (fc.pickup_km + fc.haul_km) / max(config.reposition_speed_kmpm, 0.01)
            finish_est = state.sim_minutes + SCAN_COST_MINUTES + trip_minutes_est
            # 完成时刻所在自然日的剩余时间
            finish_day_end = ((int(finish_est) // 1440) + 1) * 1440
            day_remaining = finish_day_end - finish_est
            rest_needed = config.min_continuous_rest_minutes
            if day_remaining < rest_needed and state.longest_rest_today < rest_needed:
                # 剩余时间不够 rest，按缺口比例施加罚分
                shortfall_ratio = 1.0 - day_remaining / rest_needed
                rest_penalty_per_day = config.penalty_weights.get("rest", 300)
                rest_lookahead_penalty = rest_penalty_per_day * shortfall_ratio

        # 13) 回家前瞻罚分：统一使用 TimelineProjector（含 scan_cost + 统一安全余量）
        go_home_penalty = compute_go_home_penalty_score(
            state, config, cargo, fc.pickup_km, fc.haul_km)

        # 14) 首单 deadline 跨天风险：下午接长途单导致次日首单延迟
        first_order_cross_day_penalty = 0.0
        if config.first_order_deadline_hour is not None:
            trip_minutes_est = (fc.pickup_km + fc.haul_km) / max(config.reposition_speed_kmpm, 0.01)
            finish_est = state.sim_minutes + SCAN_COST_MINUTES + trip_minutes_est
            # 如果运输跨天（卸货在次日）
            current_day = int(state.sim_minutes) // 1440
            finish_day = int(finish_est) // 1440
            if finish_day > current_day:
                # 卸货时间在次日的几点
                finish_hour_in_day = (finish_est % 1440) / 60
                # 卸货后需要时间找货接单（扫单循环约 30-60 分钟）
                effective_first_order_hour = finish_hour_in_day + 1.0
                if effective_first_order_hour > config.first_order_deadline_hour:
                    # 次日首单必定晚于 deadline
                    overshoot_hours = effective_first_order_hour - config.first_order_deadline_hour
                    penalty_per_violation = config.penalty_weights.get("first_order", 200)
                    first_order_cross_day_penalty = penalty_per_violation * min(2.0, overshoot_hours)

        # 15) 经验校准加分：基于卸货区域的历史表现
        experience_bonus = self._compute_experience_bonus(
            delivery_lat, delivery_lng, arrival_sim_min, params)

        # 综合得分
        score = (gross_income
                 - distance_cost
                 - time_cost
                 - deadhead_penalty
                 - preference_penalty
                 + position_bonus
                 - deadhead_budget_penalty
                 + home_bonus
                 + efficiency_bonus
                 - load_wait_cost
                 + first_order_urgency
                 - rest_lookahead_penalty
                 - go_home_penalty
                 - first_order_cross_day_penalty
                 + experience_bonus)

        breakdown = {
            "gross_income": gross_income,
            "distance_cost": -distance_cost,
            "time_cost": -time_cost,
            "deadhead_penalty": -deadhead_penalty,
            "preference_penalty": -preference_penalty,
            "position_bonus": position_bonus,
            "deadhead_budget_penalty": -deadhead_budget_penalty,
            "home_bonus": home_bonus,
            "efficiency_bonus": efficiency_bonus,
            "load_wait_cost": -load_wait_cost,
            "first_order_urgency": first_order_urgency,
            "rest_lookahead_penalty": -rest_lookahead_penalty,
            "go_home_penalty": -go_home_penalty,
            "first_order_cross_day_penalty": -first_order_cross_day_penalty,
            "experience_bonus": experience_bonus,
        }

        return score, breakdown

    # ------------------------------------------------------------------
    # 动态时间成本校准
    # ------------------------------------------------------------------

    def _calibrate_time_cost(self, base_cost: float, state: DriverState,
                              params: "StrategyParams") -> float:
        """根据时段、月度进度和供给密度动态校准时间机会成本。

        核心思想：时间在高峰期更值钱（货多竞争激烈），在低谷期不值钱；
        月末冲刺时时间更宝贵；供给丰富时时间成本更高（机会成本大）。
        """
        hour = state.hour_in_day()
        day = state.current_day()

        # 时段因子：高峰时间成本高，低谷时间成本低
        is_peak = any(start <= hour <= end for start, end in params.peak_hours)
        if is_peak:
            time_factor = 1.4
        elif hour >= 22 or hour <= 5:
            time_factor = 0.5  # 深夜几乎没有货，时间不值钱
        else:
            time_factor = 1.0

        # 月度进度因子：月末时间更宝贵
        if day >= 27:
            month_factor = 1.5
        elif day >= 20:
            month_factor = 1.2
        else:
            month_factor = 1.0

        # 供给密度因子：供给丰富时机会成本高
        richness = self.supply_predictor.get_supply_richness(
            state.current_lat, state.current_lng, state.sim_minutes)
        supply_factor = 1.0 + richness * 0.5  # 最高 +50%

        calibrated = base_cost * time_factor * month_factor * supply_factor
        # 限制在合理范围
        return max(0.01, min(0.25, calibrated))

    # ------------------------------------------------------------------
    # 装货时间窗感知
    # ------------------------------------------------------------------

    @staticmethod
    def _estimate_load_wait(cargo: dict[str, Any], state: DriverState,
                            pickup_km: float, config: DriverConfig) -> float:
        """估算接单后在装货点的等待时间（分钟）。

        根据 cargo['load_time'] 窗口和当前时间 + 空驶时间推算到达时刻，
        若早于装货窗口开始，差值即为等待时间。
        """
        load_time = cargo.get("load_time")
        if not load_time or not isinstance(load_time, list) or len(load_time) != 2:
            return 0.0

        try:
            load_start_min = _parse_datetime_to_sim_minutes(str(load_time[0]))
        except (ValueError, IndexError):
            return 0.0

        # 预估到达装货点的仿真时刻
        travel_minutes = pickup_km / max(config.reposition_speed_kmpm, 0.01)
        arrival_min = state.sim_minutes + travel_minutes

        if arrival_min < load_start_min:
            return load_start_min - arrival_min
        return 0.0

    def estimate_load_wait_for_cargo(self, cargo: dict[str, Any],
                                     state: DriverState, pickup_km: float,
                                     config: DriverConfig) -> float:
        """公共接口：供 ModelDecisionService 查询装货等待时间。"""
        return self._estimate_load_wait(cargo, state, pickup_km, config)

    # ------------------------------------------------------------------
    # 动态位置加分
    # ------------------------------------------------------------------

    def _enhanced_position_bonus(self, lat: float, lng: float,
                                  arrival_sim_min: int,
                                  params: "StrategyParams") -> float:
        """增强版位置优势评估：结合供给预测器和热点追踪器。

        优先使用 SupplyPredictor 的数据驱动预测（有足够历史数据时），
        fallback 到 HotspotTracker 的距离加分（初始阶段）。
        """
        # 尝试数据驱动的未来价值预测
        future_value = self.supply_predictor.predict_location_future_value(
            lat, lng, arrival_sim_min, horizon_minutes=120)

        if future_value > 0:
            # 将未来价值映射到加分：归一化到 [0, cap] 范围
            # future_value 典型范围 20~200，用 sigmoid 映射
            import math
            cap = params.position_bonus_cap * params.position_bonus_factor
            # 扩大 cap 上限：数据驱动的加分可以更大（最高 3 倍原 cap）
            enhanced_cap = cap * 3.0
            normalized = 1.0 / (1.0 + math.exp(-0.02 * (future_value - 80)))
            data_bonus = normalized * enhanced_cap

            # 如果热点数据也有，做加权融合
            hotspot_bonus = self._hotspot_distance_bonus(lat, lng, params)
            if hotspot_bonus > 0:
                return data_bonus * 0.7 + hotspot_bonus * 0.3
            return data_bonus

        # Fallback: 纯热点距离加分
        return self._hotspot_distance_bonus(lat, lng, params)

    def _hotspot_distance_bonus(self, lat: float, lng: float,
                                params: "StrategyParams") -> float:
        """原始热点距离加分（作为 fallback）。"""
        hotspots = self.hotspot_tracker.get_hotspots()
        if not hotspots:
            return 0.0

        min_dist = min(
            haversine_km(lat, lng, hs[0], hs[1])
            for hs in hotspots
        )

        cap = params.position_bonus_cap * params.position_bonus_factor
        near = params.position_near_km
        fade = params.position_fade_km

        if min_dist < near:
            return cap
        elif min_dist < fade:
            return cap * (fade - min_dist) / (fade - near)
        return 0.0

    # ------------------------------------------------------------------
    # 经验校准加分
    # ------------------------------------------------------------------

    def _compute_experience_bonus(
        self, delivery_lat: float, delivery_lng: float,
        arrival_sim_min: int, params: "StrategyParams",
    ) -> float:
        """基于卸货区域的历史经验计算校准加分。

        如果经验库显示该卸货区域在对应时段"下一单等待时间短"（好位置），
        给正向加分；如果等待时间长（差位置），给负向调整。
        """
        exp = self._current_experience
        if exp is None:
            return 0.0

        # 经验摘要来自 ModelDecisionService 注入的当前场景摘要
        # 用历史"下一单等待时间"来校准位置价值
        # 等待时间短 → 好位置 → 正向加分
        # 等待时间长 → 差位置 → 不加分或轻微扣分
        avg_wait = exp.avg_next_wait
        confidence = exp.confidence

        if confidence <= 0:
            return 0.0

        # 将等待时间映射到加分：
        # 等待 < 30min → 好位置，加分最高 20 * confidence
        # 等待 30-120min → 一般，小幅加分
        # 等待 > 120min → 差位置，轻微扣分
        cap = params.position_bonus_cap  # 复用 position_bonus_cap 作为上限参考

        if avg_wait < 30:
            # 好位置：加分 = cap * (1 - wait/30) * confidence * 经验权重
            bonus = cap * (1.0 - avg_wait / 30.0) * confidence * 0.4
        elif avg_wait < 120:
            # 一般位置：微弱正加分
            bonus = cap * 0.1 * (1.0 - (avg_wait - 30) / 90.0) * confidence * 0.4
        else:
            # 差位置：轻微扣分（不超过 -5）
            bonus = -min(5.0, (avg_wait - 120) / 60.0 * 2.0) * confidence

        return bonus

    # ------------------------------------------------------------------
    # 等待价值估算
    # ------------------------------------------------------------------

    def compute_wait_value(self, state: DriverState, config: DriverConfig,
                           avg_score_recent: float) -> float:
        """估算"等待"的期望价值，用于判断是否接当前最优单还是再等。

        简单模型：如果当前最优货源的分数低于历史均分的 70%，建议等待。
        """
        # 考虑时间的价值衰减：越接近月末/回家 deadline，等待价值越低
        hour = state.hour_in_day()
        day = state.current_day()

        # 基础等待价值 = 最近平均得分 * 到来概率
        # 经验校准：如果当前位置历史上"下一单等待长"，降低等待价值（早走为妙）
        exp = self._current_experience
        experience_adj = 1.0
        if exp is not None and exp.confidence > 0.3:
            if exp.avg_next_wait > 120:
                experience_adj = 0.6  # 差位置，少等
            elif exp.avg_next_wait < 30:
                experience_adj = 1.3  # 好位置，值得多等

        base_wait_value = avg_score_recent * 0.6 * experience_adj

        # 时段调整：高峰时段等待更有价值
        if 8 <= hour <= 11 or 14 <= hour <= 18:
            base_wait_value *= 1.2
        elif hour >= 20 or hour <= 6:
            base_wait_value *= 0.5

        # 月末压力
        if day >= 25:
            base_wait_value *= 0.7

        return base_wait_value
