"""
Support for Fibaro lights.

For more details about this platform, please refer to the documentation at
https://home-assistant.io/components/light.fibaro/
"""

import logging
import asyncio
from functools import partial

from homeassistant.components.light import (
    ATTR_BRIGHTNESS, ATTR_HS_COLOR, ATTR_WHITE_VALUE, ENTITY_ID_FORMAT,
    SUPPORT_BRIGHTNESS, SUPPORT_COLOR, SUPPORT_WHITE_VALUE, Light)
import homeassistant.util.color as color_util
from homeassistant.components.fibaro import (
    FIBARO_CONTROLLER, FIBARO_DEVICES, FibaroDevice)

_LOGGER = logging.getLogger(__name__)

DEPENDENCIES = ['fibaro']


def scaleto255(value):
    """Scale the input value from 0-100 to 0-255."""
    # Fibaro has a funny way of storing brightness either 0-100 or 0-99
    # depending on device type (e.g. dimmer vs led)
    if value > 98:
        value = 100
    return max(0, min(255, ((value * 256.0) / 100.0)))


def scaleto100(value):
    """Scale the input value from 0-255 to 0-100."""
    # Make sure a low but non-zero value is not rounded down to zero
    if 0 < value < 3:
        return 1
    return max(0, min(100, ((value * 100.4) / 255.0)))


async def async_setup_platform(hass,
                               config,
                               async_add_entities,
                               discovery_info=None):
    """Perform the setup for Fibaro controller devices."""
    if discovery_info is None:
        return

    async_add_entities(
        [FibaroLight(device, hass.data[FIBARO_CONTROLLER])
         for device in hass.data[FIBARO_DEVICES]['light']], True)


class FibaroLight(FibaroDevice, Light):
    """Representation of a Fibaro Light, including dimmable."""

    def __init__(self, fibaro_device, controller):
        """Initialize the light."""
        self._supported_flags = 0
        self._last_brightness = 0
        self._color = (0, 0)
        self._brightness = None
        self._white = 0

        self._update_lock = asyncio.Lock()
        if 'levelChange' in fibaro_device.interfaces:
            self._supported_flags |= SUPPORT_BRIGHTNESS
        if 'color' in fibaro_device.properties and \
                'setColor' in fibaro_device.actions:
            self._supported_flags |= SUPPORT_COLOR
        if 'setW' in fibaro_device.actions:
            self._supported_flags |= SUPPORT_WHITE_VALUE
        super().__init__(fibaro_device, controller)
        self.entity_id = ENTITY_ID_FORMAT.format(self.ha_id)

    @property
    def brightness(self):
        """Return the brightness of the light."""
        return scaleto255(self._brightness)

    @property
    def hs_color(self):
        """Return the color of the light."""
        return self._color

    @property
    def white_value(self):
        """Return the white value of this light between 0..255."""
        return self._white

    @property
    def supported_features(self):
        """Flag supported features."""
        return self._supported_flags

    async def async_turn_on(self, **kwargs):
        """Turn the light on."""
        async with self._update_lock:
            await self.hass.async_add_executor_job(
                partial(self._turn_on, **kwargs))

    def _turn_on(self, **kwargs):
        """Really turn the light on."""
        if self._supported_flags & SUPPORT_BRIGHTNESS:
            target_brightness = kwargs.get(ATTR_BRIGHTNESS)

            # No brightness specified, so we either restore it to
            # last brightness or switch it on at maximum level
            if target_brightness is None:
                if self._brightness == 0:
                    if self._last_brightness:
                        self._brightness = self._last_brightness
                    else:
                        self._brightness = 100
            else:
                # We set it to the target brightness and turn it on
                self._brightness = scaleto100(target_brightness)

        if self._supported_flags & SUPPORT_COLOR:
            # Update based on parameters
            self._white = kwargs.get(ATTR_WHITE_VALUE, self._white)
            self._color = kwargs.get(ATTR_HS_COLOR, self._color)
            rgb = color_util.color_hs_to_RGB(*self._color)
            self.call_set_color(
                int(rgb[0] * self._brightness / 99.0 + 0.5),
                int(rgb[1] * self._brightness / 99.0 + 0.5),
                int(rgb[2] * self._brightness / 99.0 + 0.5),
                int(self._white * self._brightness / 99.0 +
                    0.5))
            if self.state == 'off':
                self.set_level(int(self._brightness))
            return

        if self._supported_flags & SUPPORT_BRIGHTNESS:
            self.set_level(int(self._brightness))
            return

        # The simplest case is left for last. No dimming, just switch on
        self.call_turn_on()

    async def async_turn_off(self, **kwargs):
        """Turn the light off."""
        async with self._update_lock:
            await self.hass.async_add_executor_job(
                partial(self._turn_off, **kwargs))

    def _turn_off(self, **kwargs):
        """Really turn the light off."""
        # Let's save the last brightness level before we switch it off
        if (self._supported_flags & SUPPORT_BRIGHTNESS) and \
                self._brightness and self._brightness > 0:
            self._last_brightness = self._brightness
        self._brightness = 0
        self.call_turn_off()

    @property
    def is_on(self):
        """Return true if device is on."""
        return self.current_binary_state

    async def async_update(self):
        """Update the state."""
        async with self._update_lock:
            await self.hass.async_add_executor_job(self._update)

    def _update(self):
        """Really update the state."""
        # Brightness handling
        if self._supported_flags & SUPPORT_BRIGHTNESS:
            self._brightness = float(self.fibaro_device.properties.value)
        # Color handling
        if self._supported_flags & SUPPORT_COLOR and \
                'color' in self.fibaro_device.properties and \
                ',' in self.fibaro_device.properties.color:
            # Fibaro communicates the color as an 'R, G, B, W' string
            rgbw_s = self.fibaro_device.properties.color
            if rgbw_s == '0,0,0,0' and\
                    'lastColorSet' in self.fibaro_device.properties:
                rgbw_s = self.fibaro_device.properties.lastColorSet
            rgbw_list = [int(i) for i in rgbw_s.split(",")][:4]
            if rgbw_list[0] or rgbw_list[1] or rgbw_list[2]:
                self._color = color_util.color_RGB_to_hs(*rgbw_list[:3])
            if (self._supported_flags & SUPPORT_WHITE_VALUE) and \
                    self.brightness != 0:
                self._white = min(255, max(0, rgbw_list[3]*100.0 /
                                           self._brightness))
