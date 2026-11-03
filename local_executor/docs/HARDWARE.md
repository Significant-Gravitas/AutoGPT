# Hardware Access — Wire Spec

> **Status**: Draft v0.1 — spec only; handler implementation deferred.
>
> Closes VISION.md §4. The shim already detects `hardware_serial` /
> `hardware_usb` / `hardware_gpio` capabilities via `pyserial` /
> `pyusb` / `RPi.GPIO` (or `gpiozero`) presence and advertises them
> in HELLO under `capabilities[]` + populates `hardware_devices[]`.
> This doc pins the wire ops so handlers (and the platform-side
> adapter) can be written against a stable contract.

The platform-side adapter (`LocalPCShim.hardware.*` proxy in
`backend/copilot/tools/local_pc_shim.py`) plus the MCP tool
registration mirror the file/computer-use surface — one proxy method
per wire op, one MCP tool per platform-LLM use case. Permission
gating: each device-class capability (`hardware_serial`,
`hardware_usb`, `hardware_gpio`) must be in `HELLO_ACK.granted_capabilities`
before the platform may issue any op against that class. Per-device
allow-listing (e.g. "allow /dev/ttyUSB0 but not /dev/ttyUSB1") is a
v2 concern — v1 is "capability granted → all devices of that class
accessible."

---

## Device discovery

Device enumeration is part of HELLO and refreshed on demand via
`HARDWARE_LIST`. The shim does NOT poll continuously — hot-plug events
are out of scope for v1.

### `HELLO.payload.hardware_devices[]` (already specified)

```json
[
  {"type": "serial", "port": "/dev/ttyUSB0", "desc": "Arduino Uno"},
  {"type": "usb",    "vid": "2341", "pid": "0043", "desc": "Arduino Uno"},
  {"type": "gpio",   "chip": "/dev/gpiochip0", "lines": 28, "desc": "Raspberry Pi 4"}
]
```

### `HARDWARE_LIST_REQUEST` (platform → shim)

Force a re-enumeration (user just plugged in a device).

```json
{
  "type": "HARDWARE_LIST_REQUEST",
  "id": "req-uuid",
  "ts": 1234567890.0,
  "payload": {
    "type": "serial"    // "serial" | "usb" | "gpio" | null = all
  }
}
```

### `HARDWARE_LIST_RESPONSE` (shim → platform)

```json
{
  "type": "HARDWARE_LIST_RESPONSE",
  "id": "req-uuid",
  "ts": 1234567890.1,
  "payload": {
    "devices": [
      {"type": "serial", "port": "/dev/ttyUSB0", "desc": "Arduino Uno",
       "vid": "2341", "pid": "0043", "manufacturer": "Arduino LLC",
       "serial_number": "85631303731351F012E0"},
      {"type": "usb", "vid": "046d", "pid": "c52b",
       "manufacturer": "Logitech", "product": "Unifying Receiver",
       "device_class": 3}
    ]
  }
}
```

`serial_number` (USB serial descriptor — confusingly, not the same as
serial port) is the only stable identifier across reboots / re-plugs;
`port` (/dev/ttyUSB0) can shift. Platform code that needs a stable
reference between turns should remember `serial_number` and resolve to
the current `port` at use time via `HARDWARE_LIST_REQUEST`.

---

## Serial ports

`pyserial` is the canonical backend. UART over USB-serial bridges
(FT232, CH340, CP210x) all expose as `/dev/tty*` (Linux/macOS) or
`COMn` (Windows) and are handled identically.

### `SERIAL_OPEN` (platform → shim)

```json
{
  "type": "SERIAL_OPEN",
  "id": "req-uuid",
  "ts": 1234567890.0,
  "payload": {
    "port": "/dev/ttyUSB0",       // or COMn on Windows
    "baudrate": 115200,
    "bytesize": 8,                 // 5 | 6 | 7 | 8
    "parity": "none",              // "none" | "even" | "odd" | "mark" | "space"
    "stopbits": 1,                 // 1 | 1.5 | 2
    "timeout_seconds": 5.0,        // read timeout; null = block forever
    "rtscts": false,               // hardware flow control
    "dsrdtr": false,
    "xonxoff": false               // software flow control
  }
}
```

### `SERIAL_OPEN_RESPONSE` (shim → platform)

```json
{
  "type": "SERIAL_OPEN_RESPONSE",
  "id": "req-uuid",
  "ts": 1234567890.1,
  "payload": {
    "handle": "ser_<uuid>"   // opaque per-process handle; pass to read/write/close
  }
}
```

Per the same UUID-handle pattern as `WINDOW_LIST` (locked in Q2), the
handle is shim-minted, never reused, and wiped on HELLO. A dead/stale
handle in any op returns `SERIAL_STALE` (modeled on `WINDOW_STALE`).

### `SERIAL_WRITE` (platform → shim)

```json
{
  "type": "SERIAL_WRITE",
  "id": "req-uuid",
  "ts": 1234567890.0,
  "payload": {
    "handle": "ser_<uuid>",
    "data": "AAEC",              // base64-encoded bytes
    "drain": true                 // wait for the OS buffer to flush before ACK
  }
}
```

Returns `ACK` with `{bytes_written: int}`. Partial writes (rare on a
typical UART) report the actual count and the caller retries the rest.

### `SERIAL_READ` (platform → shim)

```json
{
  "type": "SERIAL_READ",
  "id": "req-uuid",
  "ts": 1234567890.0,
  "payload": {
    "handle": "ser_<uuid>",
    "size": 256,                  // max bytes to read
    "timeout_seconds": null,      // override the SERIAL_OPEN timeout for this read
    "until": null                 // optional byte pattern (base64); read up to + including
  }
}
```

### `SERIAL_READ_RESPONSE` (shim → platform)

```json
{
  "type": "SERIAL_READ_RESPONSE",
  "id": "req-uuid",
  "ts": 1234567890.1,
  "payload": {
    "data": "T0sNCg==",           // base64-encoded bytes actually read
    "timed_out": false,
    "in_waiting": 0               // bytes still buffered after this read
  }
}
```

### `SERIAL_CLOSE` (platform → shim)

```json
{
  "type": "SERIAL_CLOSE",
  "id": "req-uuid",
  "ts": 1234567890.0,
  "payload": {"handle": "ser_<uuid>"}
}
```

Returns `ACK`. The shim is responsible for closing all SERIAL handles
on WS disconnect — orphan handles leak file descriptors otherwise.

---

## USB (raw)

`pyusb` for bulk + HID transfers. The platform LLM doesn't get to
issue arbitrary USB control transfers in v1 — too easy to brick
firmware. Bulk + interrupt + HID only.

### `USB_OPEN` (platform → shim)

```json
{
  "type": "USB_OPEN",
  "id": "req-uuid",
  "ts": 1234567890.0,
  "payload": {
    "vid": "2341",                 // hex string
    "pid": "0043",
    "serial_number": null,         // null = first match
    "interface": 0,
    "detach_kernel_driver": false  // Linux: detach kernel driver (hidraw, etc.)
                                   // before claiming. Risky; refuse on Win/mac.
  }
}
```

### `USB_OPEN_RESPONSE`

```json
{
  "type": "USB_OPEN_RESPONSE",
  "id": "req-uuid",
  "ts": 1234567890.1,
  "payload": {"handle": "usb_<uuid>"}
}
```

### `USB_TRANSFER` (platform → shim)

Single envelope covers IN + OUT, bulk + interrupt.

```json
{
  "type": "USB_TRANSFER",
  "id": "req-uuid",
  "ts": 1234567890.0,
  "payload": {
    "handle": "usb_<uuid>",
    "endpoint": 1,                 // EP address (0x01 OUT, 0x81 IN, etc.)
    "direction": "out",            // "in" | "out"
    "data": "AAEC",                // base64; ignored when direction="in"
    "size": 64,                    // bytes to read when direction="in"
    "timeout_seconds": 5.0
  }
}
```

### `USB_TRANSFER_RESPONSE`

```json
{
  "type": "USB_TRANSFER_RESPONSE",
  "id": "req-uuid",
  "ts": 1234567890.1,
  "payload": {
    "data": "T0s=",                // base64-encoded bytes for direction="in"
    "bytes_transferred": 2,
    "timed_out": false
  }
}
```

### `USB_CLOSE`

```json
{
  "type": "USB_CLOSE",
  "id": "req-uuid",
  "ts": 1234567890.0,
  "payload": {"handle": "usb_<uuid>"}
}
```

---

## GPIO (Raspberry Pi / similar)

`RPi.GPIO` is deprecated upstream; `gpiozero` (which uses `lgpio` or
`rpi-lgpio`) is the recommended backend. Linux-only (`/dev/gpiochip*`).
The shim refuses these ops on macOS / Windows / WSL2 with
`FEATURE_NOT_SUPPORTED`.

### `GPIO_CONFIGURE` (platform → shim)

```json
{
  "type": "GPIO_CONFIGURE",
  "id": "req-uuid",
  "ts": 1234567890.0,
  "payload": {
    "chip": "/dev/gpiochip0",
    "line": 17,                    // BCM line number
    "direction": "out",            // "in" | "out"
    "initial_value": 0,            // 0 | 1; ignored when direction="in"
    "pull": "none",                // "none" | "up" | "down"; ignored when out
    "active_low": false
  }
}
```

Returns `ACK` with `{handle: "gpio_<uuid>"}`. Each `(chip, line)` pair
gets its own handle. Lines must be released via `GPIO_CLOSE` or shim
shutdown — Linux GPIO sysfs holds locks otherwise.

### `GPIO_WRITE` (platform → shim)

```json
{
  "type": "GPIO_WRITE",
  "id": "req-uuid",
  "ts": 1234567890.0,
  "payload": {
    "handle": "gpio_<uuid>",
    "value": 1
  }
}
```

### `GPIO_READ` (platform → shim)

```json
{
  "type": "GPIO_READ",
  "id": "req-uuid",
  "ts": 1234567890.0,
  "payload": {"handle": "gpio_<uuid>"}
}
```

### `GPIO_READ_RESPONSE`

```json
{
  "type": "GPIO_READ_RESPONSE",
  "id": "req-uuid",
  "ts": 1234567890.1,
  "payload": {"value": 1}
}
```

### `GPIO_CLOSE`

```json
{
  "type": "GPIO_CLOSE",
  "id": "req-uuid",
  "ts": 1234567890.0,
  "payload": {"handle": "gpio_<uuid>"}
}
```

### PWM (deferred to v2)

Software PWM via `gpiozero` works but timing jitter limits use cases.
Hardware PWM requires platform-specific config (Pi: BCM 12/13/18/19
only). Deferred.

---

## Error codes

Added on top of the base set in PROTOCOL.md:

- `HARDWARE_NOT_FOUND` — `port` / `(vid, pid)` / `(chip, line)` doesn't
  match any device the shim sees right now. Caller should
  `HARDWARE_LIST_REQUEST` to refresh.
- `HARDWARE_BUSY` — device is already opened by another process (or by
  the kernel — Linux). Caller can retry after the holder releases, but
  v1 has no notification when that happens.
- `HARDWARE_OPEN_FAILED` — generic open failure (permissions, broken
  cable, driver missing). Carries `details.os_error` with the raw
  errno + message.
- `SERIAL_STALE` / `USB_STALE` / `GPIO_STALE` — handle is no longer
  valid (shim restarted, device unplugged, explicit close).
- `HARDWARE_TRANSFER_FAILED` — write/read failed mid-op. Carries
  `details.bytes_partial` so the caller can resume.

---

## Per-OS support matrix

| Class | Linux | macOS | Windows | WSL2 |
|---|---|---|---|---|
| Serial (USB-UART bridges, real RS-232) | ✅ pyserial | ✅ pyserial | ✅ pyserial | ⚠️ requires usbipd-win passthrough on the host |
| Serial (Bluetooth virtual ports) | ✅ | ✅ | ✅ | ❌ |
| USB (bulk + interrupt) | ✅ pyusb + libusb | ✅ pyusb + libusb (codesign requirements; document) | ✅ pyusb + libusbK / WinUSB (driver install required) | ⚠️ usbipd-win |
| USB (HID) | ✅ hidapi or pyusb | ✅ hidapi | ✅ hidapi | ⚠️ usbipd-win |
| USB (control transfers) | ❌ deferred (firmware-brick risk) | ❌ | ❌ | ❌ |
| GPIO | ✅ gpiozero (lgpio / rpi-lgpio) on Pi-class boards | ❌ FEATURE_NOT_SUPPORTED | ❌ | ❌ |

`HELLO.capabilities` advertises only what the OS + installed libs
actually support. The platform reads the capability list and refuses
to even attempt unsupported ops, so the LLM gets a clean
`CAPABILITY_NOT_GRANTED` rather than a confusing `FEATURE_NOT_SUPPORTED`
mid-session.

---

## Security considerations

Hardware access bypasses the file-jail entirely. Three lines of defense:

1. **Capability gate at HELLO.** The user opts in at install time via
   the shim CLI: `autogpt-shim auth --scope hardware_serial,hardware_gpio`.
   No request body field can re-enable a missing scope.
2. **OS-level permissions** (Linux: `dialout` / `gpio` groups; macOS:
   driver entitlements; Windows: WinUSB driver install). The shim
   inherits the user's permissions and cannot escalate.
3. **Audit log** records every `*_OPEN` + `*_TRANSFER` with the device
   identity, byte counts (never payload content), and outcome. Per
   AUDIT_LOG.md spec.

Out of scope for v1 spec (tracked as follow-ups):
- Per-device allow-listing ("allow /dev/ttyUSB0 but not /dev/ttyUSB1").
- Bandwidth quotas (cap MB/min per device-class).
- "Confirm before each open" interactive prompt (cua-driver style).
- USB control transfers (firmware update endpoints).
- I²C / SPI / 1-Wire — different chip-level protocols; each deserves
  its own design pass.

---

## References

- VISION.md §4 — the original capability description.
- PROTOCOL.md — envelope, version negotiation, error-code base set.
- CROSS_PLATFORM.md — capability advertisement matrix this complements.
- AUDIT_LOG.md — audit record shape per op.
