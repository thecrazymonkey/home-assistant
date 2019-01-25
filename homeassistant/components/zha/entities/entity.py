"""
Entity for Zigbee Home Automation.

For more details about this component, please refer to the documentation at
https://home-assistant.io/components/zha/
"""
import asyncio
import logging
from random import uniform

from homeassistant.const import ATTR_ENTITY_ID
from homeassistant.core import callback
from homeassistant.helpers import entity
from homeassistant.helpers.device_registry import CONNECTION_ZIGBEE
from homeassistant.util import slugify
from ..const import (
    DATA_ZHA, DATA_ZHA_BRIDGE_ID, DOMAIN, ATTR_CLUSTER_ID, ATTR_ATTRIBUTE,
    ATTR_VALUE, ATTR_MANUFACTURER, ATTR_COMMAND, SERVER, ATTR_COMMAND_TYPE,
    ATTR_ARGS, IN, OUT, CLIENT_COMMANDS, SERVER_COMMANDS)
from ..helpers import bind_configure_reporting

_LOGGER = logging.getLogger(__name__)

ENTITY_SUFFIX = 'entity_suffix'


class ZhaEntity(entity.Entity):
    """A base class for ZHA entities."""

    _domain = None  # Must be overridden by subclasses

    def __init__(self, endpoint, in_clusters, out_clusters, manufacturer,
                 model, application_listener, unique_id, new_join=False,
                 **kwargs):
        """Init ZHA entity."""
        self._device_state_attributes = {}
        self._name = None
        ieee = endpoint.device.ieee
        ieeetail = ''.join(['%02x' % (o, ) for o in ieee[-4:]])
        if manufacturer and model is not None:
            self.entity_id = "{}.{}_{}_{}_{}{}".format(
                self._domain,
                slugify(manufacturer),
                slugify(model),
                ieeetail,
                endpoint.endpoint_id,
                kwargs.get(ENTITY_SUFFIX, ''),
            )
            self._name = "{} {}".format(manufacturer, model)
        else:
            self.entity_id = "{}.zha_{}_{}{}".format(
                self._domain,
                ieeetail,
                endpoint.endpoint_id,
                kwargs.get(ENTITY_SUFFIX, ''),
            )

        self._endpoint = endpoint
        self._in_clusters = in_clusters
        self._out_clusters = out_clusters
        self._new_join = new_join
        self._state = None
        self._unique_id = unique_id

        # Normally the entity itself is the listener. Sub-classes may set this
        # to a dict of cluster ID -> listener to receive messages for specific
        # clusters separately
        self._in_listeners = {}
        self._out_listeners = {}

        self._initialized = False
        self.manufacturer_code = None
        application_listener.register_entity(ieee, self)

    async def get_clusters(self):
        """Get zigbee clusters from this entity."""
        return {
            IN: self._in_clusters,
            OUT: self._out_clusters
        }

    async def _get_cluster(self, cluster_id, cluster_type=IN):
        """Get zigbee cluster from this entity."""
        if cluster_type == IN:
            cluster = self._in_clusters[cluster_id]
        else:
            cluster = self._out_clusters[cluster_id]
        if cluster is None:
            _LOGGER.warning('in_cluster with id: %s not found on entity: %s',
                            cluster_id, self.entity_id)
        return cluster

    async def get_cluster_attributes(self, cluster_id, cluster_type=IN):
        """Get zigbee attributes for specified cluster."""
        cluster = await self._get_cluster(cluster_id, cluster_type)
        if cluster is None:
            return
        return cluster.attributes

    async def write_zigbe_attribute(self, cluster_id, attribute, value,
                                    cluster_type=IN, manufacturer=None):
        """Write a value to a zigbee attribute for a cluster in this entity."""
        cluster = await self._get_cluster(cluster_id, cluster_type)
        if cluster is None:
            return

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
                self.entity_id,
                response
            )
            return response
        except DeliveryError as exc:
            _LOGGER.debug(
                'failed to set attribute: %s %s %s %s %s',
                '{}: {}'.format(ATTR_VALUE, value),
                '{}: {}'.format(ATTR_ATTRIBUTE, attribute),
                '{}: {}'.format(ATTR_CLUSTER_ID, cluster_id),
                '{}: {}'.format(ATTR_ENTITY_ID, self.entity_id),
                exc
            )

    async def get_cluster_commands(self, cluster_id, cluster_type=IN):
        """Get zigbee commands for specified cluster."""
        cluster = await self._get_cluster(cluster_id, cluster_type)
        if cluster is None:
            return
        return {
            CLIENT_COMMANDS: cluster.client_commands,
            SERVER_COMMANDS: cluster.server_commands,
        }

    async def issue_cluster_command(self, cluster_id, command, command_type,
                                    args, cluster_type=IN,
                                    manufacturer=None):
        """Issue a command against specified zigbee cluster on this entity."""
        cluster = await self._get_cluster(cluster_id, cluster_type)
        if cluster is None:
            return
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
            '{}: {}'.format(ATTR_ENTITY_ID, self.entity_id)
        )
        return response

    async def async_added_to_hass(self):
        """Handle entity addition to hass.

        It is now safe to update the entity state
        """
        for cluster_id, cluster in self._in_clusters.items():
            cluster.add_listener(self._in_listeners.get(cluster_id, self))
        for cluster_id, cluster in self._out_clusters.items():
            cluster.add_listener(self._out_listeners.get(cluster_id, self))

        self._endpoint.device.zdo.add_listener(self)

        if self._new_join:
            self.hass.async_create_task(self.async_configure())

        self._initialized = True

    async def async_configure(self):
        """Set cluster binding and attribute reporting."""
        for cluster_key, attrs in self.zcl_reporting_config.items():
            cluster = self._get_cluster_from_report_config(cluster_key)
            if cluster is None:
                continue

            manufacturer = None
            if cluster.cluster_id >= 0xfc00 and self.manufacturer_code:
                manufacturer = self.manufacturer_code

            skip_bind = False  # bind cluster only for the 1st configured attr
            for attr, details in attrs.items():
                min_report_interval, max_report_interval, change = details
                await bind_configure_reporting(
                    self.entity_id, cluster, attr,
                    min_report=min_report_interval,
                    max_report=max_report_interval,
                    reportable_change=change,
                    skip_bind=skip_bind,
                    manufacturer=manufacturer
                )
                skip_bind = True
                await asyncio.sleep(uniform(0.1, 0.5))
        _LOGGER.debug("%s: finished configuration", self.entity_id)

    def _get_cluster_from_report_config(self, cluster_key):
        """Parse an entry from zcl_reporting_config dict."""
        from zigpy.zcl import Cluster as Zcl_Cluster

        cluster = None
        if isinstance(cluster_key, Zcl_Cluster):
            cluster = cluster_key
        elif isinstance(cluster_key, str):
            cluster = getattr(self._endpoint, cluster_key, None)
        elif isinstance(cluster_key, int):
            if cluster_key in self._in_clusters:
                cluster = self._in_clusters[cluster_key]
            elif cluster_key in self._out_clusters:
                cluster = self._out_clusters[cluster_key]
        elif issubclass(cluster_key, Zcl_Cluster):
            cluster_id = cluster_key.cluster_id
            if cluster_id in self._in_clusters:
                cluster = self._in_clusters[cluster_id]
            elif cluster_id in self._out_clusters:
                cluster = self._out_clusters[cluster_id]
        return cluster

    @property
    def name(self):
        """Return Entity's default name."""
        return self._name

    @property
    def zcl_reporting_config(self):
        """Return a dict of ZCL attribute reporting configuration.

        {
            Cluster_Class: {
                attr_id: (min_report_interval, max_report_interval, change),
                attr_name: (min_rep_interval, max_rep_interval, change)
            }
            Cluster_Instance: {
                attr_id: (min_report_interval, max_report_interval, change),
                attr_name: (min_rep_interval, max_rep_interval, change)
            }
            cluster_id: {
                attr_id: (min_report_interval, max_report_interval, change),
                attr_name: (min_rep_interval, max_rep_interval, change)
            }
            'cluster_name': {
                attr_id: (min_report_interval, max_report_interval, change),
                attr_name: (min_rep_interval, max_rep_interval, change)
            }
        }
        """
        return {}

    @property
    def unique_id(self) -> str:
        """Return a unique ID."""
        return self._unique_id

    @property
    def device_state_attributes(self):
        """Return device specific state attributes."""
        return self._device_state_attributes

    @property
    def should_poll(self) -> bool:
        """Let ZHA handle polling."""
        return False

    @callback
    def attribute_updated(self, attribute, value):
        """Handle an attribute updated on this cluster."""
        pass

    @callback
    def zdo_command(self, tsn, command_id, args):
        """Handle a ZDO command received on this cluster."""
        pass

    @callback
    def device_announce(self, device):
        """Handle device_announce zdo event."""
        self.async_schedule_update_ha_state(force_refresh=True)

    @callback
    def permit_duration(self, permit_duration):
        """Handle permit_duration zdo event."""
        pass

    @property
    def device_info(self):
        """Return a device description for device registry."""
        ieee = str(self._endpoint.device.ieee)
        return {
            'connections': {(CONNECTION_ZIGBEE, ieee)},
            'identifiers': {(DOMAIN, ieee)},
            ATTR_MANUFACTURER: self._endpoint.manufacturer,
            'model': self._endpoint.model,
            'name': self.name or ieee,
            'via_hub': (DOMAIN, self.hass.data[DATA_ZHA][DATA_ZHA_BRIDGE_ID]),
        }

    @callback
    def zha_send_event(self, cluster, command, args):
        """Relay entity events to hass."""
        pass  # don't relay events from entities
