"""
Support for the Fibaro devices.

For more details about this platform, please refer to the documentation.
https://home-assistant.io/components/fibaro/
"""

import logging
from collections import defaultdict
import voluptuous as vol

from homeassistant.const import (ATTR_ARMED, ATTR_BATTERY_LEVEL,
                                 CONF_PASSWORD, CONF_URL, CONF_USERNAME,
                                 EVENT_HOMEASSISTANT_STOP)
import homeassistant.helpers.config_validation as cv
from homeassistant.util import convert, slugify
from homeassistant.helpers import discovery
from homeassistant.helpers.entity import Entity

REQUIREMENTS = ['fiblary3==0.1.7']

_LOGGER = logging.getLogger(__name__)
DOMAIN = 'fibaro'
FIBARO_DEVICES = 'fibaro_devices'
FIBARO_CONTROLLER = 'fibaro_controller'
ATTR_CURRENT_POWER_W = "current_power_w"
ATTR_CURRENT_ENERGY_KWH = "current_energy_kwh"
CONF_PLUGINS = "plugins"

FIBARO_COMPONENTS = ['binary_sensor', 'cover', 'light', 'sensor', 'switch']

FIBARO_TYPEMAP = {
    'com.fibaro.multilevelSensor': "sensor",
    'com.fibaro.binarySwitch': 'switch',
    'com.fibaro.multilevelSwitch': 'switch',
    'com.fibaro.FGD212': 'light',
    'com.fibaro.FGR': 'cover',
    'com.fibaro.doorSensor': 'binary_sensor',
    'com.fibaro.doorWindowSensor': 'binary_sensor',
    'com.fibaro.FGMS001': 'binary_sensor',
    'com.fibaro.heatDetector': 'binary_sensor',
    'com.fibaro.lifeDangerSensor': 'binary_sensor',
    'com.fibaro.smokeSensor': 'binary_sensor',
    'com.fibaro.remoteSwitch': 'switch',
    'com.fibaro.sensor': 'sensor',
    'com.fibaro.colorController': 'light'
}

CONFIG_SCHEMA = vol.Schema({
    DOMAIN: vol.Schema({
        vol.Required(CONF_PASSWORD): cv.string,
        vol.Required(CONF_USERNAME): cv.string,
        vol.Required(CONF_URL): cv.url,
        vol.Optional(CONF_PLUGINS, default=False): cv.boolean,
    })
}, extra=vol.ALLOW_EXTRA)


class FibaroController():
    """Initiate Fibaro Controller Class."""

    _room_map = None            # Dict for mapping roomId to room object
    _device_map = None          # Dict for mapping deviceId to device object
    fibaro_devices = None       # List of devices by type
    _callbacks = {}             # Dict of update value callbacks by deviceId
    _client = None               # Fiblary's Client object for communication
    _state_handler = None        # Fiblary's StateHandler object
    _import_plugins = None      # Whether to import devices from plugins

    def __init__(self, username, password, url, import_plugins):
        """Initialize the Fibaro controller."""
        from fiblary3.client.v4.client import Client as FibaroClient
        self._client = FibaroClient(url, username, password)

    def connect(self):
        """Start the communication with the Fibaro controller."""
        try:
            login = self._client.login.get()
        except AssertionError:
            _LOGGER.error("Can't connect to Fibaro HC. "
                          "Please check URL.")
            return False
        if login is None or login.status is False:
            _LOGGER.error("Invalid login for Fibaro HC. "
                          "Please check username and password.")
            return False

        self._room_map = {room.id: room for room in self._client.rooms.list()}
        self._read_devices()
        return True

    def enable_state_handler(self):
        """Start StateHandler thread for monitoring updates."""
        from fiblary3.client.v4.client import StateHandler
        self._state_handler = StateHandler(self._client, self._on_state_change)

    def disable_state_handler(self):
        """Stop StateHandler thread used for monitoring updates."""
        self._state_handler.stop()
        self._state_handler = None

    def _on_state_change(self, state):
        """Handle change report received from the HomeCenter."""
        callback_set = set()
        for change in state.get('changes', []):
            dev_id = change.pop('id')
            for property_name, value in change.items():
                if property_name == 'log':
                    if value and value != "transfer OK":
                        _LOGGER.debug("LOG %s: %s",
                                      self._device_map[dev_id].friendly_name,
                                      value)
                    continue
                if property_name == 'logTemp':
                    continue
                if property_name in self._device_map[dev_id].properties:
                    self._device_map[dev_id].properties[property_name] = \
                        value
                    _LOGGER.debug("<- %s.%s = %s",
                                  self._device_map[dev_id].ha_id,
                                  property_name,
                                  str(value))
                else:
                    _LOGGER.warning("Error updating %s data of %s, not found",
                                    property_name,
                                    self._device_map[dev_id].ha_id)
                if dev_id in self._callbacks:
                    callback_set.add(dev_id)
        for item in callback_set:
            self._callbacks[item]()

    def register(self, device_id, callback):
        """Register device with a callback for updates."""
        self._callbacks[device_id] = callback

    @staticmethod
    def _map_device_to_type(device):
        """Map device to HA device type."""
        # Use our lookup table to identify device type
        device_type = FIBARO_TYPEMAP.get(
            device.type, FIBARO_TYPEMAP.get(device.baseType))

        # We can also identify device type by its capabilities
        if device_type is None:
            if 'setBrightness' in device.actions:
                device_type = 'light'
            elif 'turnOn' in device.actions:
                device_type = 'switch'
            elif 'open' in device.actions:
                device_type = 'cover'
            elif 'value' in device.properties:
                if device.properties.value in ('true', 'false'):
                    device_type = 'binary_sensor'
                else:
                    device_type = 'sensor'

        # Switches that control lights should show up as lights
        if device_type == 'switch' and \
                'isLight' in device.properties and \
                device.properties.isLight == 'true':
            device_type = 'light'
        return device_type

    def _read_devices(self):
        """Read and process the device list."""
        devices = self._client.devices.list()
        self._device_map = {}
        for device in devices:
            if device.roomID == 0:
                room_name = 'Unknown'
            else:
                room_name = self._room_map[device.roomID].name
            device.friendly_name = room_name + ' ' + device.name
            device.ha_id = '{}_{}_{}'.format(
                slugify(room_name), slugify(device.name), device.id)
            self._device_map[device.id] = device
        self.fibaro_devices = defaultdict(list)
        for device in self._device_map.values():
            if device.enabled and \
                    (not device.isPlugin or self._import_plugins):
                device.mapped_type = self._map_device_to_type(device)
                if device.mapped_type:
                    self.fibaro_devices[device.mapped_type].append(device)
                else:
                    _LOGGER.debug("%s (%s, %s) not mapped",
                                  device.ha_id, device.type,
                                  device.baseType)


def setup(hass, config):
    """Set up the Fibaro Component."""
    hass.data[FIBARO_CONTROLLER] = controller = \
        FibaroController(config[DOMAIN][CONF_USERNAME],
                         config[DOMAIN][CONF_PASSWORD],
                         config[DOMAIN][CONF_URL],
                         config[DOMAIN][CONF_PLUGINS])

    def stop_fibaro(event):
        """Stop Fibaro Thread."""
        _LOGGER.info("Shutting down Fibaro connection")
        hass.data[FIBARO_CONTROLLER].disable_state_handler()

    if controller.connect():
        hass.data[FIBARO_DEVICES] = controller.fibaro_devices
        for component in FIBARO_COMPONENTS:
            discovery.load_platform(hass, component, DOMAIN, {}, config)
        controller.enable_state_handler()
        hass.bus.listen_once(EVENT_HOMEASSISTANT_STOP, stop_fibaro)
        return True

    return False


class FibaroDevice(Entity):
    """Representation of a Fibaro device entity."""

    def __init__(self, fibaro_device, controller):
        """Initialize the device."""
        self.fibaro_device = fibaro_device
        self.controller = controller
        self._name = fibaro_device.friendly_name
        self.ha_id = fibaro_device.ha_id

    async def async_added_to_hass(self):
        """Call when entity is added to hass."""
        self.controller.register(self.fibaro_device.id, self._update_callback)

    def _update_callback(self):
        """Update the state."""
        self.schedule_update_ha_state(True)

    @property
    def level(self):
        """Get the level of Fibaro device."""
        if 'value' in self.fibaro_device.properties:
            return self.fibaro_device.properties.value
        return None

    @property
    def level2(self):
        """Get the tilt level of Fibaro device."""
        if 'value2' in self.fibaro_device.properties:
            return self.fibaro_device.properties.value2
        return None

    def dont_know_message(self, action):
        """Make a warning in case we don't know how to perform an action."""
        _LOGGER.warning("Not sure how to setValue: %s "
                        "(available actions: %s)", str(self.ha_id),
                        str(self.fibaro_device.actions))

    def set_level(self, level):
        """Set the level of Fibaro device."""
        self.action("setValue", level)
        if 'value' in self.fibaro_device.properties:
            self.fibaro_device.properties.value = level
        if 'brightness' in self.fibaro_device.properties:
            self.fibaro_device.properties.brightness = level

    def set_level2(self, level):
        """Set the level2 of Fibaro device."""
        self.action("setValue2", level)
        if 'value2' in self.fibaro_device.properties:
            self.fibaro_device.properties.value2 = level

    def call_turn_on(self):
        """Turn on the Fibaro device."""
        self.action("turnOn")

    def call_turn_off(self):
        """Turn off the Fibaro device."""
        self.action("turnOff")

    def call_set_color(self, red, green, blue, white):
        """Set the color of Fibaro device."""
        color_str = "{},{},{},{}".format(int(red), int(green),
                                         int(blue), int(white))
        self.fibaro_device.properties.color = color_str
        self.action("setColor", str(int(red)), str(int(green)),
                    str(int(blue)), str(int(white)))

    def action(self, cmd, *args):
        """Perform an action on the Fibaro HC."""
        if cmd in self.fibaro_device.actions:
            getattr(self.fibaro_device, cmd)(*args)
            _LOGGER.debug("-> %s.%s%s called", str(self.ha_id),
                          str(cmd), str(args))
        else:
            self.dont_know_message(cmd)

    @property
    def hidden(self) -> bool:
        """Return True if the entity should be hidden from UIs."""
        return self.fibaro_device.visible is False

    @property
    def current_power_w(self):
        """Return the current power usage in W."""
        if 'power' in self.fibaro_device.properties:
            power = self.fibaro_device.properties.power
            if power:
                return convert(power, float, 0.0)
        else:
            return None

    @property
    def current_binary_state(self):
        """Return the current binary state."""
        if self.fibaro_device.properties.value == 'false':
            return False
        if self.fibaro_device.properties.value == 'true' or \
                int(self.fibaro_device.properties.value) > 0:
            return True
        return False

    @property
    def name(self):
        """Return the name of the device."""
        return self._name

    @property
    def should_poll(self):
        """Get polling requirement from fibaro device."""
        return False

    def update(self):
        """Call to update state."""
        pass

    @property
    def device_state_attributes(self):
        """Return the state attributes of the device."""
        attr = {}

        try:
            if 'battery' in self.fibaro_device.interfaces:
                attr[ATTR_BATTERY_LEVEL] = \
                    int(self.fibaro_device.properties.batteryLevel)
            if 'fibaroAlarmArm' in self.fibaro_device.interfaces:
                attr[ATTR_ARMED] = bool(self.fibaro_device.properties.armed)
            if 'power' in self.fibaro_device.interfaces:
                attr[ATTR_CURRENT_POWER_W] = convert(
                    self.fibaro_device.properties.power, float, 0.0)
            if 'energy' in self.fibaro_device.interfaces:
                attr[ATTR_CURRENT_ENERGY_KWH] = convert(
                    self.fibaro_device.properties.energy, float, 0.0)
        except (ValueError, KeyError):
            pass

        attr['id'] = self.ha_id
        return attr
