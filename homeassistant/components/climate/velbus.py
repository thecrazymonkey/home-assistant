"""
Support for Velbus thermostat.

For more details about this platform, please refer to the documentation
https://home-assistant.io/components/climate.velbus/
"""
import logging

from homeassistant.components.climate import (
    SUPPORT_OPERATION_MODE, SUPPORT_TARGET_TEMPERATURE, ClimateDevice)
from homeassistant.components.velbus import (
    DOMAIN as VELBUS_DOMAIN, VelbusEntity)
from homeassistant.const import ATTR_TEMPERATURE

_LOGGER = logging.getLogger(__name__)

DEPENDENCIES = ['velbus']

OPERATION_LIST = ['comfort', 'day', 'night', 'safe']

SUPPORT_FLAGS = (SUPPORT_TARGET_TEMPERATURE | SUPPORT_OPERATION_MODE)


async def async_setup_platform(
        hass, config, async_add_entities, discovery_info=None):
    """Set up the Velbus thermostat platform."""
    if discovery_info is None:
        return

    sensors = []
    for sensor in discovery_info:
        module = hass.data[VELBUS_DOMAIN].get_module(sensor[0])
        channel = sensor[1]
        sensors.append(VelbusClimate(module, channel))

    async_add_entities(sensors)


class VelbusClimate(VelbusEntity, ClimateDevice):
    """Representation of a Velbus thermostat."""

    @property
    def supported_features(self):
        """Return the list off supported features."""
        return SUPPORT_FLAGS

    @property
    def temperature_unit(self):
        """Return the unit this state is expressed in."""
        return self._module.get_unit(self._channel)

    @property
    def current_temperature(self):
        """Return the current temperature."""
        return self._module.get_state(self._channel)

    @property
    def current_operation(self):
        """Return current operation ie. heat, cool, idle."""
        return self._module.get_climate_mode()

    @property
    def operation_list(self):
        """Return the list of available operation modes."""
        return OPERATION_LIST

    @property
    def target_temperature(self):
        """Return the temperature we try to reach."""
        return self._module.get_climate_target()

    def set_operation_mode(self, operation_mode):
        """Set new target operation mode."""
        self._module.set_mode(operation_mode)
        self.schedule_update_ha_state()

    def set_temperature(self, **kwargs):
        """Set new target temperatures."""
        if kwargs.get(ATTR_TEMPERATURE) is not None:
            self._module.set_temp(kwargs.get(ATTR_TEMPERATURE))
            self.schedule_update_ha_state()
