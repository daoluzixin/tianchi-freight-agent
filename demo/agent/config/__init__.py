"""配置模块：司机配置、偏好解析。"""

from agent.config.driver_config import (
    DriverConfig, QuietWindow, FamilyEvent, SpecialCargo,
    get_config, register_config, clear_configs, build_config_from_parsed,
)
from agent.config.preference_parser import PreferenceParser, ParsedPreferences
