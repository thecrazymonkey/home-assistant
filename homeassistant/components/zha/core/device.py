"""
Device for Zigbee Home Automation.

For more details about this component, please refer to the documentation at
https://home-assistant.io/components/zha/
"""
import asyncio
import logging

from homeassistant.helpers.dispatcher import (
    async_dispatcher_connect, async_dispatcher_send
)
from .const import (
    ATTR_MANUFACTURER, LISTENER_BATTERY, SIGNAL_AVAILABLE, IN, OUT,
    ATTR_CLUSTER_ID, ATTR_ATTRIBUTE, ATTR_VALUE, ATTR_COMMAND, SERVER,
    ATTR_COMMAND_TYPE, ATTR_ARGS, CLIENT_COMMANDS, SERVER_COMMANDS,
    ATTR_ENDPOINT_ID, IEEE, MODEL, NAME, UNKNOWN
)
from .listeners import EventRelayListener

_LOGGER = logging.getLogger(__name__)


class ZHADevice:
    """ZHA Zigbee device object."""

    def __init__(self, hass, zigpy_device, zha_gateway):
        """Initialize the gateway."""
        self.hass = hass
        self._zigpy_device = zigpy_device
        # Get first non ZDO endpoint id to use to get manufacturer and model
        endpoint_ids = zigpy_device.endpoints.keys()
        self._manufacturer = UNKNOWN
        self._model = UNKNOWN
        ept_id = next((ept_id for ept_id in endpoint_ids if ept_id != 0), None)
        if ept_id is not None:
            self._manufacturer = zigpy_device.endpoints[ept_id].manufacturer
            self._model = zigpy_device.endpoints[ept_id].model
        self._zha_gateway = zha_gateway
        self.cluster_listeners = {}
        self._relay_listeners = []
        self._all_listeners = []
        self._name = "{} {}".format(
            self.manufacturer,
            self.model
        )
        self._available = False
        self._available_signal = "{}_{}_{}".format(
            self.name, self.ieee, SIGNAL_AVAILABLE)
        self._unsub = async_dispatcher_connect(
            self.hass,
            self._available_signal,
            self.async_initialize
        )

    @property
    def name(self):
        """Return device name."""
        return self._name

    @property
    def ieee(self):
        """Return ieee address for device."""
        return self._zigpy_device.ieee

    @property
    def manufacturer(self):
        """Return ieee address for device."""
        return self._manufacturer

    @property
    def model(self):
        """Return ieee address for device."""
        return self._model

    @property
    def nwk(self):
        """Return nwk for device."""
        return self._zigpy_device.nwk

    @property
    def lqi(self):
        """Return lqi for device."""
        return self._zigpy_device.lqi

    @property
    def rssi(self):
        """Return rssi for device."""
        return self._zigpy_device.rssi

    @property
    def last_seen(self):
        """Return last_seen for device."""
        return self._zigpy_device.last_seen

    @property
    def manufacturer_code(self):
        """Return manufacturer code for device."""
        # will eventually get this directly from Zigpy
        return None

    @property
    def gateway(self):
        """Return the gateway for this device."""
        return self._zha_gateway

    @property
    def all_listeners(self):
        """Return cluster listeners and relay listeners for device."""
        return self._all_listeners

    @property
    def available_signal(self):
        """Signal to use to subscribe to device availability changes."""
        return self._available_signal

    @property
    def available(self):
        """Return True if sensor is available."""
        return self._available

    def update_available(self, available):
        """Set sensor availability."""
        if self._available != available and available:
            # Update the state the first time the device comes online
            async_dispatcher_send(
                self.hass,
                self._available_signal,
                False
            )
            async_dispatcher_send(
                self.hass,
                "{}_{}".format(self._available_signal, 'entity'),
                True
            )
        self._available = available

    @property
    def device_info(self):
        """Return a device description for device."""
        ieee = str(self.ieee)
        return {
            IEEE: ieee,
            ATTR_MANUFACTURER: self.manufacturer,
            MODEL: self.model,
            NAME: self.name or ieee
        }

    def add_cluster_listener(self, cluster_listener):
        """Add cluster listener to device."""
        # only keep 1 power listener
        if cluster_listener.name is LISTENER_BATTERY and \
                LISTENER_BATTERY in self.cluster_listeners:
            return
        self._all_listeners.append(cluster_listener)
        if isinstance(cluster_listener, EventRelayListener):
            self._relay_listeners.append(cluster_listener)
        else:
            self.cluster_listeners[cluster_listener.name] = cluster_listener

    async def async_configure(self):
        """Configure the device."""
        _LOGGER.debug('%s: started configuration', self.name)
        await self._execute_listener_tasks('async_configure')
        _LOGGER.debug('%s: completed configuration', self.name)

    async def async_initialize(self, from_cache):
        """Initialize listeners."""
        _LOGGER.debug('%s: started initialization', self.name)
        await self._execute_listener_tasks('async_initialize', from_cache)
        _LOGGER.debug('%s: completed initialization', self.name)

    async def _execute_listener_tasks(self, task_name, *args):
        """Gather and execute a set of listener tasks."""
        listener_tasks = []
        for listener in self.all_listeners:
            listener_tasks.append(
                self._async_create_task(listener, task_name, *args))
        await asyncio.gather(*listener_tasks)

    async def _async_create_task(self, listener, func_name, *args):
        """Configure a single listener on this device."""
        try:
            await getattr(listener, func_name)(*args)
            _LOGGER.debug('%s: listener: %s %s stage succeeded',
                          self.name,
                          "{}-{}".format(
                              listener.name, listener.unique_id),
                          func_name)
        except Exception as ex:  # pylint: disable=broad-except
            _LOGGER.warning(
                '%s listener: %s %s stage failed ex: %s',
                self.name,
                "{}-{}".format(listener.name, listener.unique_id),
                func_name,
                ex
            )

    async def async_unsub_dispatcher(self):
        """Unsubscribe the dispatcher."""
        if self._unsub:
            self._unsub()

    async def get_clusters(self):
        """Get all clusters for this device."""
        return {
            ep_id: {
                IN: endpoint.in_clusters,
                OUT: endpoint.out_clusters
            } for (ep_id, endpoint) in self._zigpy_device.endpoints.items()
            if ep_id != 0
        }

    async def get_cluster(self, endpooint_id, cluster_id, cluster_type=IN):
        """Get zigbee cluster from this entity."""
        clusters = await self.get_clusters()
        return clusters[endpooint_id][cluster_type][cluster_id]

    async def get_cluster_attributes(self, endpooint_id, cluster_id,
                                     cluster_type=IN):
        """Get zigbee attributes for specified cluster."""
        cluster = await self.get_cluster(endpooint_id, cluster_id,
                                         cluster_type)
        if cluster is None:
            return None
        return cluster.attributes

    async def get_cluster_commands(self, endpooint_id, cluster_id,
                                   cluster_type=IN):
        """Get zigbee commands for specified cluster."""
        cluster = await self.get_cluster(endpooint_id, cluster_id,
                                         cluster_type)
        if cluster is None:
            return None
        return {
            CLIENT_COMMANDS: cluster.client_commands,
            SERVER_COMMANDS: cluster.server_commands,
        }

    async def write_zigbee_attribute(self, endpooint_id, cluster_id,
                                     attribute, value, cluster_type=IN,
                                     manufacturer=None):
        """Write a value to a zigbee attribute for a cluster in this entity."""
        cluster = await self.get_cluster(
            endpooint_id, cluster_id, cluster_type)
        if cluster is None:
            return None

        from zigpy.exceptions import DeliveryError
        try:
            response = await cluster.write_attributes(
                {attribute: value},
                manufacturer=manufacturer
            )
            _LOGGER.debug(
                'set: %s for attr: %s to cluster: %s for entity: %s - res: %s',
                value,
                attribute,
                cluster_id,
                endpooint_id,
                response
            )
            return response
        except DeliveryError as exc:
            _LOGGER.debug(
                'failed to set attribute: %s %s %s %s %s',
                '{}: {}'.format(ATTR_VALUE, value),
                '{}: {}'.format(ATTR_ATTRIBUTE, attribute),
                '{}: {}'.format(ATTR_CLUSTER_ID, cluster_id),
                '{}: {}'.format(ATTR_ENDPOINT_ID, endpooint_id),
                exc
            )
            return None

    async def issue_cluster_command(self, endpooint_id, cluster_id, command,
                                    command_type, args, cluster_type=IN,
                                    manufacturer=None):
        """Issue a command against specified zigbee cluster on this entity."""
        cluster = await self.get_cluster(
            endpooint_id, cluster_id, cluster_type)
        if cluster is None:
            return None
        response = None
        if command_type == SERVER:
            response = await cluster.command(command, *args,
                                             manufacturer=manufacturer,
                                             expect_reply=True)
        else:
            response = await cluster.client_command(command, *args)

        _LOGGER.debug(
            'Issued cluster command: %s %s %s %s %s %s %s',
            '{}: {}'.format(ATTR_CLUSTER_ID, cluster_id),
            '{}: {}'.format(ATTR_COMMAND, command),
            '{}: {}'.format(ATTR_COMMAND_TYPE, command_type),
            '{}: {}'.format(ATTR_ARGS, args),
            '{}: {}'.format(ATTR_CLUSTER_ID, cluster_type),
            '{}: {}'.format(ATTR_MANUFACTURER, manufacturer),
            '{}: {}'.format(ATTR_ENDPOINT_ID, endpooint_id)
        )
        return response
