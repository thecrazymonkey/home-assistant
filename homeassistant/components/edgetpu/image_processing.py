"""Support for performing EdgeTPU classification on images."""
import logging

import voluptuous as vol

from homeassistant.components.image_processing import (
    CONF_ENTITY_ID, CONF_NAME, CONF_SOURCE, PLATFORM_SCHEMA,
    ImageProcessingEntity)
from homeassistant.core import split_entity_id
from homeassistant.helpers import template
import homeassistant.helpers.config_validation as cv

_LOGGER = logging.getLogger(__name__)

ATTR_MATCHES = 'matches'
ATTR_SUMMARY = 'summary'
ATTR_TOTAL_MATCHES = 'total_matches'

CONF_CATEGORIES = 'categories'
CONF_CATEGORY = 'category'
CONF_FILE_OUT = 'file_out'
CONF_PATH = 'path'
CONF_LABELS = 'labels'
CONF_MODEL = 'model'
CONF_MODEL_DIR = 'model_dir'
CONF_TPU_DEVICE = 'device'
CONF_TPU_TOP_K = 'top_k'
CONF_TPU_THRESHOLD = 'threshold'
CONF_TPU_KEEP_ASPECT_RATIO = 'keep_aspect_ratio'
CONF_TPU_RESAMPLE = 'resample'

DEFAULT_THRESHOLD = 0.05
DEFAULT_TOP_K = 10
DEFAULT_KEEP_ASPECT_RATIO = True
DEFAULT_RESAMPLE = 0

PLATFORM_SCHEMA = PLATFORM_SCHEMA.extend({
    vol.Optional(CONF_FILE_OUT, default=[]):
        vol.All(cv.ensure_list, [cv.template]),
    vol.Required(CONF_MODEL): vol.Schema({
        vol.Required(CONF_PATH): cv.isfile,
        vol.Optional(CONF_LABELS): cv.isfile,
        vol.Optional(CONF_MODEL_DIR): cv.isdir,
        vol.Optional(CONF_TPU_DEVICE): cv.string,
        vol.Optional(CONF_TPU_THRESHOLD, default=DEFAULT_THRESHOLD): cv.small_float,
        vol.Optional(CONF_TPU_KEEP_ASPECT_RATIO, default=DEFAULT_KEEP_ASPECT_RATIO): cv.boolean,
        vol.Optional(CONF_TPU_RESAMPLE, default=DEFAULT_RESAMPLE): cv.positive_int,
        vol.Optional(CONF_TPU_TOP_K, default=DEFAULT_TOP_K): cv.positive_int,
    })
})

def setup_platform(hass, config, add_entities, discovery_info=None):
    """Set up the EdgeTPU image processing platform."""

    try:
        # Verify that the TensorFlow Object Detection API is pre-installed
        # pylint: disable=unused-import,unused-variable
        from edgetpu.detection.engine import DetectionEngine
    except ImportError:
        # pylint: disable=line-too-long
        _LOGGER.error(
            "No EdgeTPU Object Detection library found! Install or compile ") # noqa
        return

    entities = []

    for camera in config[CONF_SOURCE]:
        entities.append(EdgeTPUImageProcessor(
            hass, camera[CONF_ENTITY_ID], camera.get(CONF_NAME),
            config))

    add_entities(entities)

class EdgeTPUImageProcessor(ImageProcessingEntity):
    """Representation of an EdgeTPU image processor."""

    def __init__(self, hass, camera_entity, name, config):
        """Initialize the EdgeTPU entity."""
        from edgetpu.detection.engine import DetectionEngine # pylint: disable=import-error
        model_config = config.get(CONF_MODEL)
        _LOGGER.info("config = %s", model_config)
        self.hass = hass
        self._camera_entity = camera_entity
        _LOGGER.info("camera = %s", self._camera_entity)
        if name:
            self._name = name
        else:
            self._name = "EdgeTPU {0}".format(
                split_entity_id(camera_entity)[1])
        self._file_out = config.get(CONF_FILE_OUT)
        self._model = model_config.get(CONF_PATH)
        self._threshold = model_config.get(CONF_TPU_THRESHOLD)
        self._top_k = model_config.get(CONF_TPU_TOP_K)
        self._keep_aspect_ratio = model_config.get(CONF_TPU_KEEP_ASPECT_RATIO)
        self._resample = model_config.get(CONF_TPU_RESAMPLE)
        self._engine = DetectionEngine(self._model, device_path=model_config.get(CONF_TPU_DEVICE))
        labels = model_config.get(CONF_LABELS)
        self._labels = self._read_label_file(labels) if labels else None

        template.attach(hass, self._file_out)

        self._matches = {}
        self._total_matches = 0
        self._last_image = None

    @property
    def camera_entity(self):
        """Return camera entity id from process pictures."""
        return self._camera_entity

    @property
    def name(self):
        """Return the name of the image processor."""
        return self._name

    @property
    def state(self):
        """Return the state of the entity."""
        return self._total_matches

    @property
    def device_state_attributes(self):
        """Return device specific state attributes."""
        return {
            ATTR_MATCHES: self._matches,
#            ATTR_SUMMARY: {item: len(values)
#                           for item, values in self._matches.items()},
            ATTR_TOTAL_MATCHES: self._total_matches
        }

    # Function to read labels from text files.
    def _read_label_file(self, file_path):
        with open(file_path, 'r', encoding="utf-8") as source_file:
            lines = source_file.readlines()
        ret = {}
        for line in lines:
            pair = line.strip().split(maxsplit=1)
            ret[int(pair[0])] = pair[1].strip()
        return ret

    def process_image(self, image):
        """Process the image."""
        from PIL import Image
        from PIL import ImageDraw
        _LOGGER.debug("Model=%s", self._model)

        matches = {}
        total_matches = 0

        # Open image.
#        _LOGGER.info("image = %s", image)
        import io
        img = Image.open(io.BytesIO(bytearray(image)))
#        img.save("/tmp/test.jpg")
        draw = ImageDraw.Draw(img)

        # Run inference.
        ans = self._engine.DetectWithImage(img, threshold=self._threshold,
                                           keep_aspect_ratio=self._keep_aspect_ratio,
                                           relative_coord=False, top_k=self._top_k)
        # Display result.
        if ans:
            for obj in ans:
                _LOGGER.info("obj = %s", obj)
                if self._labels:
                    _LOGGER.info(self._labels[obj.label_id])
                _LOGGER.info("score = %f", obj.score)
                box = obj.bounding_box.flatten().tolist()
                _LOGGER.info("box = %s", box)
                # Draw a rectangle.
                draw.rectangle(box, outline='red')
                if self._file_out:
                    for path_template in self._file_out:
                        if isinstance(path_template, template.Template):
                            img.save(path_template.render(
                                camera_entity=self._camera_entity))
                        else:
                            img.save(path_template)
                if 'Face' not in matches.keys():
                    matches['Face'] = []
                matches['Face'].append({
                    'score': float(obj.score),
                    'box': box
                })
                total_matches += 1
        else:
            _LOGGER.info("No object detected!")

        self._matches = matches
        self._total_matches = total_matches
