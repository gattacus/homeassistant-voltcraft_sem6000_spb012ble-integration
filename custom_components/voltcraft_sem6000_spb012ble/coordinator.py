from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass

from bleak import BleakClient, BleakGATTCharacteristic
from bleak.exc import BleakError

from homeassistant.core import HomeAssistant
from homeassistant.helpers.device_registry import CONNECTION_BLUETOOTH, DeviceInfo
from homeassistant.helpers.device_registry import format_mac
from homeassistant.helpers.update_coordinator import DataUpdateCoordinator, UpdateFailed

from .const import COMMAND_UUID, DEVICE_NAME, DOMAIN, NOTIFY_UUID, SCAN_INTERVAL
from .protocol import (
    Command,
    MeasureNotifyPayload,
    NotifyPayload,
    SwitchModes,
    SwitchNotifyPayload,
    LoginMode,
)

_LOGGER = logging.getLogger(__name__)


@dataclass
class VoltcraftData:
    """Data from Voltcraft device measurements."""

    is_on: bool
    power: float  # Watts (converted from mW)
    voltage: float  # Volts
    current: float  # Amps (converted from mA)
    frequency: int  # Hz
    power_factor: float | None  # 0.0 - 1.0, calculated from P/(V*I)
    consumed_energy: float  # kWh (converted from Wh)

    @staticmethod
    def from_payload(payload: MeasureNotifyPayload) -> VoltcraftData:
        power = payload.power / 1000.0  # mW to W
        voltage = float(payload.voltage)
        current = payload.current / 1000.0  # mA to A

        # Power factor - calculate from P / (V * I)
        apparent_power = voltage * current
        power_factor: float | None
        if apparent_power > 0:
            power_factor = min(power / apparent_power, 1.0)
        else:
            power_factor = None

        return VoltcraftData(
            is_on=payload.is_on,
            power=power,
            voltage=voltage,
            current=current,
            frequency=payload.frequency,
            power_factor=power_factor,
            consumed_energy=payload.consumed_energy / 1000.0,  # Wh to kWh
        )


class VoltcraftDataUpdateCoordinator(DataUpdateCoordinator[VoltcraftData | None]):
    def __init__(
        self,
        hass: HomeAssistant,
        client: BleakClient,
        mac: str,
        device_name: str | None,
    ) -> None:
        super().__init__(
            hass,
            _LOGGER,
            name=f"{DOMAIN}_{mac}",
            update_interval=SCAN_INTERVAL,
        )
        self.client = client
        self.mac = format_mac(mac)
        self._device_name = device_name
        self._latest_data: VoltcraftData | None = None
        self._command_lock = asyncio.Lock()
        self._login_result_future: asyncio.Future[bool] | None = None
        self._pin_reset_attempted = False

    @property
    def device_info(self) -> DeviceInfo:
        return DeviceInfo(
            connections={(CONNECTION_BLUETOOTH, self.mac)},
            identifiers={(DOMAIN, self.mac)},
            name=self._device_name or DEVICE_NAME,
        )

    async def async_setup(self) -> None:
        await self.client.start_notify(NOTIFY_UUID, self._handle_notify)
        # Login is required for some firmware versions before control commands work.
        await asyncio.sleep(2.0)
        await self._async_login()

    async def async_shutdown(self) -> None:
        try:
            await self.client.stop_notify(NOTIFY_UUID)
        except BleakError as err:
            _LOGGER.debug("Error stopping notifications: %s", err)

        try:
            await self.client.disconnect()
        except BleakError as err:
            _LOGGER.debug("Error disconnecting client: %s", err)

    async def _async_update_data(self) -> VoltcraftData | None:
        """Fetch data from the device.

        This sends a measure command and returns the latest data.
        The actual data update happens asynchronously via a notification handler.
        """
        try:
            async with self._command_lock:
                await self._async_ensure_connected()
                async with asyncio.timeout(5.0):
                    await self.client.write_gatt_char(
                        COMMAND_UUID, Command.MEASURE.build_payload(), response=True
                    )
                    # Wait for notification to be processed.
                    await asyncio.sleep(0.5)
        except Exception as err:
            await self._async_force_disconnect()
            raise UpdateFailed(f"Failed to send measure command: {err}") from err

        return self._latest_data

    async def _handle_notify(self, sender: BleakGATTCharacteristic, data: bytearray) -> None:
        """Handle notifications from the device."""
        _LOGGER.debug("Received notification: %s", data.hex())
        
        # Login response payload: status 0 means authenticated; non-zero means rejected.
        if len(data) > 4 and data[2] == Command.LOGIN:
            status = data[4]
            accepted = status == 0
            if accepted:
                _LOGGER.debug("Voltcraft login accepted")
            else:
                _LOGGER.warning("Voltcraft login rejected with status %s", status)

            future = self._login_result_future
            if future is not None and not future.done():
                future.set_result(accepted)
            return

        payload = NotifyPayload.from_payload(data)

        match payload:
            case MeasureNotifyPayload():
                self._latest_data = VoltcraftData.from_payload(payload)
                self.async_set_updated_data(self._latest_data)

            case SwitchNotifyPayload():
                # Switch state changed, trigger immediate measure to update data
                self.hass.create_task(self.async_request_refresh())

            case None:
                _LOGGER.warning("Unknown payload received: %s", data.hex())

    async def _async_write_login_payload(self, payload: bytearray) -> bool:
        future = asyncio.get_running_loop().create_future()
        self._login_result_future = future
        try:
            await self.client.write_gatt_char(COMMAND_UUID, payload, response=True)
            return await asyncio.wait_for(future, timeout=2.0)
        except TimeoutError:
            _LOGGER.warning("Timed out waiting for Voltcraft login response")
            return False
        finally:
            if self._login_result_future is future:
                self._login_result_future = None

    async def _async_login(self) -> bool:
        """Authenticate with the device for firmware that requires a PIN login."""
        return await self._async_write_login_payload(LoginMode.build_payload())

    async def _async_reset_pin_to_default(self) -> bool:
        """Reset the device PIN to 0000 using the firmware reset command."""
        _LOGGER.warning("Resetting Voltcraft PIN to 0000 after login rejection")
        return await self._async_write_login_payload(LoginMode.build_reset_payload())

    async def _async_ensure_connected(self) -> None:
        """Reconnect and authenticate if the BLE connection was lost."""
        if self.client.is_connected:
            return

        _LOGGER.debug("Reconnecting to Voltcraft device %s", self.mac)
        await self.client.connect()
        await self.client.start_notify(NOTIFY_UUID, self._handle_notify)
        await self._async_login()
        self._latest_data = None

    async def _async_force_disconnect(self) -> None:
        """Force a clean BLE state so the next operation reconnects."""
        try:
            await self.client.disconnect()
        except BleakError as err:
            _LOGGER.debug("Error forcing disconnect: %s", err)

    async def async_send_switch_command(self, mode: SwitchModes) -> None:
        """Send a switch command to the device."""
        try:
            async with self._command_lock:
                await self._async_ensure_connected()
                async with asyncio.timeout(5.0):
                    # Some firmware accepts measurements without auth but requires
                    # authentication before relay-control writes.
                    if not await self._async_login():
                        if self._pin_reset_attempted:
                            raise BleakError("Voltcraft login rejected")

                        self._pin_reset_attempted = True
                        if not await self._async_reset_pin_to_default():
                            raise BleakError("Voltcraft PIN reset was rejected")
                        if not await self._async_login():
                            raise BleakError("Voltcraft login rejected after PIN reset")

                    payload = mode.build_payload()
                    _LOGGER.debug(
                        "Sending switch command with response: %s payload=%s",
                        mode.name,
                        payload.hex(),
                    )
                    await self.client.write_gatt_char(COMMAND_UUID, payload, response=True)
                    await asyncio.sleep(0.5)
        except Exception as err:
            await self._async_force_disconnect()
            _LOGGER.error("Failed to send switch command: %s", err)
            raise

        await self.async_request_refresh()
