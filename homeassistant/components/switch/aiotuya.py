"""
Simple platform to control **SOME** Tuya switch devices.

For more details about this platform, please refer to the documentation at
https://home-assistant.io/components/switch.tuya/
"""
import logging
import asyncio
import voluptuous as vol
from homeassistant.components.switch import SwitchDevice, PLATFORM_SCHEMA
from homeassistant.const import (CONF_NAME, CONF_HOST, CONF_ID, CONF_SWITCHES,
                                 CONF_FRIENDLY_NAME, CONF_SCAN_INTERVAL)
from homeassistant.core import callback
import homeassistant.helpers.config_validation as cv
from homeassistant.helpers.event import async_call_later

CONF_DEVICE_ID = 'device_id'
CONF_LOCAL_KEY = 'local_key'

DEFAULT_ID = '1'
DEFAULT_NAME = 'aiotuya'
# hack to load base package
REQUIREMENTS = ['aiotuya==0.7.1']

SWITCH_SCHEMA = vol.Schema({
    vol.Optional(CONF_ID, default=DEFAULT_ID): cv.string,
    vol.Optional(CONF_FRIENDLY_NAME): cv.string,
})

PLATFORM_SCHEMA = PLATFORM_SCHEMA.extend({
    vol.Optional(CONF_NAME): cv.string,
    vol.Optional(CONF_HOST, default=''): cv.string,
    vol.Required(CONF_DEVICE_ID): cv.string,
    vol.Optional(CONF_SCAN_INTERVAL, default=25): cv.time_period,
    vol.Required(CONF_LOCAL_KEY): cv.string,
    vol.Optional(CONF_ID, default=DEFAULT_ID): cv.string,
    vol.Optional(CONF_SWITCHES, default={}):
        vol.Schema({cv.slug: SWITCH_SCHEMA}),
})

_LOGGER = logging.getLogger(__name__)

async def async_setup_platform(hass, config, async_add_devices, discovery_info=None):
    """Set up of the Tuya switch."""
    from aiotuya import OutletDevice, resolveId
    devices = config.get(CONF_SWITCHES)

    _LOGGER.debug("Starting Aiotuya setup")
#    if config.get(CONF_HOST) == None:
#        parser = aiotuya.MessageParser()
#        ipResolver = TuyaIPDiscovery(config.get(CONF_DEVICE_ID), parser)
#        coro = hass.loop.create_datagram_endpoint(
#            lambda: ipResolver, local_addr=('255.255.255.255', 6666))
#        ipResolver.task = hass.async_add_job(coro)
    devices = config.get(CONF_SWITCHES)
    switches = []
    if config.get(CONF_HOST) == '':
        _LOGGER.debug("Starting IP Discovery")
        address = await hass.async_add_executor_job(resolveId, config.get(CONF_DEVICE_ID))
        _LOGGER.debug("Aiotuya discovered at %s", address)
        config[CONF_HOST] = address
    tuyamanger = TuyaManager(hass, async_add_devices)
    outlet_device = OutletDevice(
        hass.loop,
        tuyamanger,
        config.get(CONF_DEVICE_ID),
        config.get(CONF_LOCAL_KEY),
        config.get(CONF_HOST)
    )

    for object_id, device_config in devices.items():
        tuyadevice = TuyaPlug(
            outlet_device,
            device_config.get(CONF_FRIENDLY_NAME, object_id),
            device_config.get(CONF_ID),
        )
        switches.append(tuyadevice)
        _LOGGER.debug("async_setup_platform adding %s", config.get(CONF_ID))

    name = config.get(CONF_NAME)
    if name:
        tuyadevice = TuyaPlug(
            outlet_device,
            name,
            config.get(CONF_ID)
        )
        switches.append(tuyadevice)
        _LOGGER.debug("async_setup_platform adding name %s/%s",
                      config.get(CONF_NAME),
                      config.get(CONF_ID))

    tuyamanger.switches = switches
    outlet_device.tuyadevice = outlet_device
    coro = hass.loop.create_connection(lambda: outlet_device,
                                       config.get(CONF_HOST), 6668)
    hass.async_add_job(coro)
    _LOGGER.debug("Starting Aiotuya setup - end")

    return True

class TuyaIPDiscovery(asyncio.DatagramProtocol):
    """
    Represents an object for initial Tuya device IP address discovery using UDP broadcast.
    """
    def __init__(self, dev_id, parser):
        """
        Represents an object for initial Tuya device IP address discovery using UDP broadcast.

        Args:
            dev_id (str): The device id.
            parser (object): Parser
        """
        self.dev_id = dev_id
        self.parser = parser
        self.address = None
        self.task = None
        self.transport = None

    def connection_made(self, transport):
        self.transport = transport
        _LOGGER.debug("Starting Discovery UDP server")

    def datagram_received(self, data, addr):
        (error, result, command) = self.parser.extract_payload(data)

        if error is False:
            _LOGGER.debug('Resolve string=%s (command:%i', result, command)
            thisid = result['gwId']
            # check if already registered, if not add to the list and create a handler instance
            if thisid == self.dev_id:
                _LOGGER.debug('Discovered=%s on IP=%s', thisid, result['ip'])
                self.transport.close()

    def connection_lost(self, exc):
        _LOGGER.debug("Disconnected %s", exc)
        self.task.cancel()


class TuyaManager:
    """Helper class to manage Tuya Device within Hass."""

    def __init__(self, hass, add_entities):
        """
        Represents an object for handling Tuya device events and applying them onto Hass actions.

        Args:
            hass: hass object.
            add_entities: hass add_entities procedure
        """
        self.switches = []
        self.add_entities = add_entities
        self.hass = hass

    @callback
    def on_init(self):
        """Called on first reported connecto from aiotuya."""
        _LOGGER.debug('on_init()')
        self.add_entities(self.switches, update_before_add=True)
        _LOGGER.debug('on_init() - end')

    @callback
    def register_job(self, job, delay=None):
        """Helper proc to schedule new job within hass."""
        _LOGGER.debug('register_job()')
        #  not clear yet
        if delay is None:
            self.hass.async_add_job(job)
        else:
            async_call_later(self.hass, delay, job)

    @callback
    def data_parsed(self, result):
        """Called when Tuya device receives some structured data."""
        _LOGGER.debug('data_parsed(): %s', result)
        dps = result['dps']
        for switch in iter(self.switches):
            if switch.switchid in dps:
                _LOGGER.debug('Switch set: %s to %s', switch.switchid, dps[switch.switchid])
                if switch.is_on != dps[switch.switchid]:
                    switch.set_state(dps[switch.switchid])
                    # notify hass to update state
                    switch.schedule_update_ha_state()
    @callback
    def on_connection_lost(self, exc):
        """Called when Tuya device lost connection."""
        _LOGGER.debug('on_connection_lost(): %s', exc)
        # may need some handling if message did not get sent

class TuyaPlug(SwitchDevice):
    """Representation of a Tuya switch."""

    def __init__(self, device, name, switchid):
        """Initialize the Tuya switch."""
        self._device = device
        self._name = name
        self._state = False
        self._switchid = switchid

    @property
    def name(self):
        """Get name of Tuya switch."""
        return self._name

    @property
    def switchid(self):
        """Get id of Tuya sub-switch."""
        return self._switchid

    @property
    def is_on(self):
        """Check if Tuya switch is on."""
        return self._state

    @property
    def should_poll(self):
        """Async doesn't need poll.
        Need explicitly call schedule_update_ha_state() after state changed.
        """
        return False

    def set_state(self, state):
        """Set state during async update."""
        self._state = state

    async def async_turn_on(self, **kwargs):
        """Turn Tuya switch on."""
        _LOGGER.debug("TuyaPlug:async_turn_on()")
        await self._device.set_status(True, self._switchid)


    async def async_turn_off(self, **kwargs):
        """Turn Tuya switch off."""
        _LOGGER.debug("TuyaPlug:async_turn_off()")
        await self._device.set_status(False, self._switchid)

    async def async_update(self):
        """Get state of Tuya switch."""
        # should come automatically
        _LOGGER.debug("TuyaPlug:async_update()")
        await self._device.status()
        async_call_later(self.hass, 25, self.send_echo)

    async def send_echo(self, _):
        """Send echo command and set timer for new one."""
        # should come automatically
        _LOGGER.debug("TuyaPlug:send_echo()")
        await self._device.echo()
        async_call_later(self.hass, 25, self.send_echo)
