"""Device tracker platform that adds support for OwnTracks over MQTT."""
import logging

from homeassistant.core import callback
from homeassistant.components.device_tracker.const import (
    DOMAIN, SOURCE_TYPE_GPS)
from homeassistant.components.device_tracker.config_entry import (
    DeviceTrackerEntity
)
from .const import (
    DOMAIN as MA_DOMAIN,

    ATTR_ALTITUDE,
    ATTR_BATTERY,
    ATTR_COURSE,
    ATTR_DEVICE_ID,
    ATTR_DEVICE_NAME,
    ATTR_GPS_ACCURACY,
    ATTR_GPS,
    ATTR_LOCATION_NAME,
    ATTR_SPEED,
    ATTR_VERTICAL_ACCURACY,

    SIGNAL_LOCATION_UPDATE,
)
from .helpers import device_info

_LOGGER = logging.getLogger(__name__)


async def async_setup_entry(hass, entry, async_add_entities):
    """Set up OwnTracks based off an entry."""
    @callback
    def _receive_data(data):
        """Receive set location."""
        dev_id = entry.data[ATTR_DEVICE_ID]
        device = hass.data[MA_DOMAIN][DOMAIN].get(dev_id)

        if device is not None:
            device.update_data(data)
            return

        device = hass.data[MA_DOMAIN][DOMAIN][dev_id] = MobileAppEntity(
            entry, data
        )
        async_add_entities([device])

    hass.helpers.dispatcher.async_dispatcher_connect(
        SIGNAL_LOCATION_UPDATE.format(entry.entry_id), _receive_data)
    return True


class MobileAppEntity(DeviceTrackerEntity):
    """Represent a tracked device."""

    def __init__(self, entry, data):
        """Set up OwnTracks entity."""
        self._entry = entry
        self._data = data

    @property
    def unique_id(self):
        """Return the unique ID."""
        return self._entry.data[ATTR_DEVICE_ID]

    @property
    def battery_level(self):
        """Return the battery level of the device."""
        return self._data.get(ATTR_BATTERY)

    @property
    def device_state_attributes(self):
        """Return device specific attributes."""
        attrs = {}
        for key in (ATTR_ALTITUDE, ATTR_COURSE,
                    ATTR_SPEED, ATTR_VERTICAL_ACCURACY):
            value = self._data.get(key)
            if value is not None:
                attrs[key] = value

        return attrs

    @property
    def location_accuracy(self):
        """Return the gps accuracy of the device."""
        return self._data.get(ATTR_GPS_ACCURACY)

    @property
    def latitude(self):
        """Return latitude value of the device."""
        gps = self._data.get(ATTR_GPS)

        if gps is None:
            return None

        return gps[0]

    @property
    def longitude(self):
        """Return longitude value of the device."""
        gps = self._data.get(ATTR_GPS)

        if gps is None:
            return None

        return gps[1]

    @property
    def location_name(self):
        """Return a location name for the current location of the device."""
        return self._data.get(ATTR_LOCATION_NAME)

    @property
    def name(self):
        """Return the name of the device."""
        return self._entry.data[ATTR_DEVICE_NAME]

    @property
    def should_poll(self):
        """No polling needed."""
        return False

    @property
    def source_type(self):
        """Return the source type, eg gps or router, of the device."""
        return SOURCE_TYPE_GPS

    @property
    def device_info(self):
        """Return the device info."""
        return device_info(self._entry.data)

    @callback
    def update_data(self, data):
        """Mark the device as seen."""
        self._data = data
        self.async_write_ha_state()
