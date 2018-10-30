"""The tests for the Jewish calendar sensor platform."""
import unittest
from datetime import time
from datetime import datetime as dt
from unittest.mock import patch

from homeassistant.util.async_ import run_coroutine_threadsafe
from homeassistant.util.dt import get_time_zone, set_default_time_zone
from homeassistant.setup import setup_component
from homeassistant.components.sensor.jewish_calendar import JewishCalSensor
from tests.common import get_test_home_assistant


class TestJewishCalenderSensor(unittest.TestCase):
    """Test the Jewish Calendar sensor."""

    TEST_LATITUDE = 31.778
    TEST_LONGITUDE = 35.235

    def setUp(self):
        """Set up things to run when tests begin."""
        self.hass = get_test_home_assistant()

    def tearDown(self):
        """Stop everything that was started."""
        self.hass.stop()
        # Reset the default timezone, so we don't affect other tests
        set_default_time_zone(get_time_zone('UTC'))

    def test_jewish_calendar_min_config(self):
        """Test minimum jewish calendar configuration."""
        config = {
            'sensor': {
                'platform': 'jewish_calendar'
            }
        }
        assert setup_component(self.hass, 'sensor', config)

    def test_jewish_calendar_hebrew(self):
        """Test jewish calendar sensor with language set to hebrew."""
        config = {
            'sensor': {
                'platform': 'jewish_calendar',
                'language': 'hebrew',
            }
        }

        assert setup_component(self.hass, 'sensor', config)

    def test_jewish_calendar_multiple_sensors(self):
        """Test jewish calendar sensor with multiple sensors setup."""
        config = {
            'sensor': {
                'platform': 'jewish_calendar',
                'sensors': [
                    'date', 'weekly_portion', 'holiday_name',
                    'holyness', 'first_light', 'gra_end_shma',
                    'mga_end_shma', 'plag_mincha', 'first_stars'
                ]
            }
        }

        assert setup_component(self.hass, 'sensor', config)

    def test_jewish_calendar_sensor_date_output(self):
        """Test Jewish calendar sensor date output."""
        test_time = dt(2018, 9, 3)
        set_default_time_zone(get_time_zone('UTC'))
        self.hass.config.latitude = self.TEST_LATITUDE
        self.hass.config.longitude = self.TEST_LONGITUDE
        sensor = JewishCalSensor(
            name='test', language='english', sensor_type='date',
            latitude=self.TEST_LATITUDE, longitude=self.TEST_LONGITUDE,
            timezone=get_time_zone("UTC"), diaspora=False)
        sensor.hass = self.hass
        with patch('homeassistant.util.dt.now', return_value=test_time):
            run_coroutine_threadsafe(
                sensor.async_update(),
                self.hass.loop).result()
            assert sensor.state == '23 Elul 5778'

    def test_jewish_calendar_sensor_date_output_hebrew(self):
        """Test Jewish calendar sensor date output in hebrew."""
        test_time = dt(2018, 9, 3)
        set_default_time_zone(get_time_zone('UTC'))
        self.hass.config.latitude = self.TEST_LATITUDE
        self.hass.config.longitude = self.TEST_LONGITUDE
        sensor = JewishCalSensor(
            name='test', language='hebrew', sensor_type='date',
            latitude=self.TEST_LATITUDE, longitude=self.TEST_LONGITUDE,
            timezone=get_time_zone("UTC"), diaspora=False)
        sensor.hass = self.hass
        with patch('homeassistant.util.dt.now', return_value=test_time):
            run_coroutine_threadsafe(
                sensor.async_update(), self.hass.loop).result()
            assert sensor.state == "כ\"ג באלול ה\' תשע\"ח"

    def test_jewish_calendar_sensor_holiday_name(self):
        """Test Jewish calendar sensor holiday name output in hebrew."""
        test_time = dt(2018, 9, 10)
        set_default_time_zone(get_time_zone('UTC'))
        self.hass.config.latitude = self.TEST_LATITUDE
        self.hass.config.longitude = self.TEST_LONGITUDE
        sensor = JewishCalSensor(
            name='test', language='hebrew', sensor_type='holiday_name',
            latitude=self.TEST_LATITUDE, longitude=self.TEST_LONGITUDE,
            timezone=get_time_zone("UTC"), diaspora=False)
        sensor.hass = self.hass
        with patch('homeassistant.util.dt.now', return_value=test_time):
            run_coroutine_threadsafe(
                sensor.async_update(), self.hass.loop).result()
            assert sensor.state == "א\' ראש השנה"

    def test_jewish_calendar_sensor_holiday_name_english(self):
        """Test Jewish calendar sensor holiday name output in english."""
        test_time = dt(2018, 9, 10)
        set_default_time_zone(get_time_zone('UTC'))
        self.hass.config.latitude = self.TEST_LATITUDE
        self.hass.config.longitude = self.TEST_LONGITUDE
        sensor = JewishCalSensor(
            name='test', language='english', sensor_type='holiday_name',
            latitude=self.TEST_LATITUDE, longitude=self.TEST_LONGITUDE,
            timezone=get_time_zone("UTC"), diaspora=False)
        sensor.hass = self.hass
        with patch('homeassistant.util.dt.now', return_value=test_time):
            run_coroutine_threadsafe(
                sensor.async_update(), self.hass.loop).result()
            assert sensor.state == "Rosh Hashana I"

    def test_jewish_calendar_sensor_holyness(self):
        """Test Jewish calendar sensor holyness value."""
        test_time = dt(2018, 9, 10)
        set_default_time_zone(get_time_zone('UTC'))
        self.hass.config.latitude = self.TEST_LATITUDE
        self.hass.config.longitude = self.TEST_LONGITUDE
        sensor = JewishCalSensor(
            name='test', language='hebrew', sensor_type='holyness',
            latitude=self.TEST_LATITUDE, longitude=self.TEST_LONGITUDE,
            timezone=get_time_zone("UTC"), diaspora=False)
        sensor.hass = self.hass
        with patch('homeassistant.util.dt.now', return_value=test_time):
            run_coroutine_threadsafe(
                sensor.async_update(), self.hass.loop).result()
            assert sensor.state == 1

    def test_jewish_calendar_sensor_torah_reading(self):
        """Test Jewish calendar sensor torah reading in hebrew."""
        test_time = dt(2018, 9, 8)
        set_default_time_zone(get_time_zone('UTC'))
        self.hass.config.latitude = self.TEST_LATITUDE
        self.hass.config.longitude = self.TEST_LONGITUDE
        sensor = JewishCalSensor(
            name='test', language='hebrew', sensor_type='weekly_portion',
            latitude=self.TEST_LATITUDE, longitude=self.TEST_LONGITUDE,
            timezone=get_time_zone("UTC"), diaspora=False)
        sensor.hass = self.hass
        with patch('homeassistant.util.dt.now', return_value=test_time):
            run_coroutine_threadsafe(
                sensor.async_update(), self.hass.loop).result()
            assert sensor.state == "פרשת נצבים"

    def test_jewish_calendar_sensor_first_stars_ny(self):
        """Test Jewish calendar sensor first stars time in NY, US."""
        test_time = dt(2018, 9, 8)
        set_default_time_zone(get_time_zone('America/New_York'))
        self.hass.config.latitude = 40.7128
        self.hass.config.longitude = -74.0060
        # self.hass.config.time_zone = get_time_zone("America/New_York")
        sensor = JewishCalSensor(
            name='test', language='hebrew', sensor_type='first_stars',
            latitude=40.7128, longitude=-74.0060,
            timezone=get_time_zone("America/New_York"), diaspora=False)
        sensor.hass = self.hass
        with patch('homeassistant.util.dt.now', return_value=test_time):
            run_coroutine_threadsafe(
                sensor.async_update(), self.hass.loop).result()
            assert sensor.state == time(19, 48)

    def test_jewish_calendar_sensor_first_stars_jerusalem(self):
        """Test Jewish calendar sensor first stars time in Jerusalem, IL."""
        set_default_time_zone(get_time_zone('Asia/Jerusalem'))
        test_time = dt(2018, 9, 8)
        self.hass.config.latitude = self.TEST_LATITUDE
        self.hass.config.longitude = self.TEST_LONGITUDE
        sensor = JewishCalSensor(
            name='test', language='hebrew', sensor_type='first_stars',
            latitude=self.TEST_LATITUDE, longitude=self.TEST_LONGITUDE,
            timezone=get_time_zone("Asia/Jerusalem"), diaspora=False)
        sensor.hass = self.hass
        with patch('homeassistant.util.dt.now', return_value=test_time):
            run_coroutine_threadsafe(
                sensor.async_update(), self.hass.loop).result()
            assert sensor.state == time(19, 21)

    def test_jewish_calendar_sensor_torah_reading_weekday(self):
        """Test the sensor showing torah reading also on weekdays."""
        set_default_time_zone(get_time_zone('Asia/Jerusalem'))
        test_time = dt(2018, 10, 14)
        self.hass.config.latitude = self.TEST_LATITUDE
        self.hass.config.longitude = self.TEST_LONGITUDE
        sensor = JewishCalSensor(
            name='test', language='hebrew', sensor_type='weekly_portion',
            latitude=self.TEST_LATITUDE, longitude=self.TEST_LONGITUDE,
            timezone=get_time_zone("Asia/Jerusalem"), diaspora=False)
        sensor.hass = self.hass
        with patch('homeassistant.util.dt.now', return_value=test_time):
            run_coroutine_threadsafe(
                sensor.async_update(), self.hass.loop).result()
            assert sensor.state == "פרשת לך לך"

    def test_jewish_calendar_sensor_date_before_sunset(self):
        """Test the sensor showing the correct date before sunset."""
        tz = get_time_zone('Asia/Jerusalem')
        set_default_time_zone(tz)
        test_time = tz.localize(dt(2018, 10, 14, 17, 0, 0))
        self.hass.config.latitude = self.TEST_LATITUDE
        self.hass.config.longitude = self.TEST_LONGITUDE
        sensor = JewishCalSensor(
            name='test', language='hebrew', sensor_type='date',
            latitude=self.TEST_LATITUDE, longitude=self.TEST_LONGITUDE,
            timezone=get_time_zone("Asia/Jerusalem"), diaspora=False)
        sensor.hass = self.hass
        with patch('homeassistant.util.dt.now', return_value=test_time):
            run_coroutine_threadsafe(
                sensor.async_update(), self.hass.loop).result()
            assert sensor.state == "ה\' בחשון ה\' תשע\"ט"

    def test_jewish_calendar_sensor_date_after_sunset(self):
        """Test the sensor showing the correct date after sunset."""
        tz = get_time_zone('Asia/Jerusalem')
        set_default_time_zone(tz)
        test_time = tz.localize(dt(2018, 10, 14, 19, 0, 0))
        self.hass.config.latitude = self.TEST_LATITUDE
        self.hass.config.longitude = self.TEST_LONGITUDE
        sensor = JewishCalSensor(
            name='test', language='hebrew', sensor_type='date',
            latitude=self.TEST_LATITUDE, longitude=self.TEST_LONGITUDE,
            timezone=get_time_zone("Asia/Jerusalem"), diaspora=False)
        sensor.hass = self.hass
        with patch('homeassistant.util.dt.now', return_value=test_time):
            run_coroutine_threadsafe(
                sensor.async_update(), self.hass.loop).result()
            assert sensor.state == "ו\' בחשון ה\' תשע\"ט"
