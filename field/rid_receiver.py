#!/usr/bin/env python3
"""AeroFence field node: log broadcast Remote ID (ASTM F3411) from Wi-Fi and Bluetooth.

Runs on a Raspberry Pi 4/5 with
  * a USB Wi-Fi adapter that supports monitor mode (e.g. MediaTek MT7612U / MT7921AU),
  * the built-in Bluetooth radio (Bluetooth legacy advertising),
  * a GPS receiver disciplining the system clock through gpsd + chrony (see README).

Every decoded message is written as one CSV row with the receiver's own timestamp, so
the fusion server (or the offline analysis) can compute time-to-first-fix, latency,
packet reception and position error.

Usage (as root):
    python3 rid_receiver.py --node N1 --iface wlan1 --channel 6 --out /home/pi/rid_logs
    python3 rid_receiver.py --node N1 --no-wifi          # Bluetooth only
    python3 rid_receiver.py --node N1 --no-ble           # Wi-Fi only
"""
from __future__ import annotations

import argparse
import asyncio
import csv
import os
import struct
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

ODID_MSG_SIZE = 25
ODID_PACK_MAX = 10
ASTM_OUI = bytes.fromhex("FA0BBC")       # ASTM vendor OUI for Wi-Fi beacon Remote ID
ASTM_OUI_TYPE = 0x0D
NAN_OUI = bytes.fromhex("506F9A")        # Wi-Fi Alliance OUI (NAN service discovery)
BLE_UUID16 = 0xFFFA                      # ASTM Remote ID service data UUID
BLE_APP_CODE = 0x0D

FIELDS = ["node", "rx_time_unix", "rx_time_utc", "bearer", "src_mac", "rssi_dbm", "channel",
          "msg_counter", "msg_type", "uas_id", "id_type", "ua_type", "status", "lat", "lon",
          "alt_geo_m", "alt_baro_m", "height_m", "speed_h_ms", "speed_v_ms", "direction_deg",
          "h_acc", "v_acc", "rid_ts_tenths", "rid_ts_unix", "operator_lat", "operator_lon"]


# --------------------------------------------------------------------------- decoding
def _rid_time_to_unix(tenths: int, rx_unix: float) -> float | None:
    """Location timestamps are tenths of a second since the start of the current UTC hour."""
    if tenths == 0xFFFF or tenths > 36000:
        return None
    hour0 = rx_unix - (rx_unix % 3600)
    t = hour0 + tenths / 10.0
    if t - rx_unix > 1800:      # message from the previous hour
        t -= 3600
    elif rx_unix - t > 1800:
        t += 3600
    return t


def decode_message(m: bytes, rx_unix: float) -> dict:
    """Decode one 25-byte Open Drone ID message (ASTM F3411-22a layout)."""
    if len(m) < ODID_MSG_SIZE:
        return {}
    mtype = m[0] >> 4
    out = {"msg_type": mtype}
    if mtype == 0x0:     # Basic ID
        out["id_type"] = m[1] >> 4
        out["ua_type"] = m[1] & 0x0F
        out["uas_id"] = m[2:22].split(b"\x00")[0].decode("ascii", "replace").strip()
    elif mtype == 0x1:   # Location / vector
        flags = m[1]
        out["status"] = flags >> 4
        ew = (flags >> 1) & 1
        mult = flags & 1
        direction = m[2] + (180 if ew else 0)
        sp = m[3]
        speed = sp * 0.25 if mult == 0 else sp * 0.75 + 255 * 0.25
        vs = struct.unpack("<b", m[4:5])[0] * 0.5
        lat, lon = struct.unpack("<ii", m[5:13])
        alt_baro, alt_geo, height = struct.unpack("<HHH", m[13:19])
        acc = m[19]
        ts = struct.unpack("<H", m[21:23])[0]
        out.update({
            "direction_deg": direction if direction <= 360 else None,
            "speed_h_ms": speed if sp != 255 else None,
            "speed_v_ms": vs if m[4] != 126 else None,    # 63 m/s encodes "unknown"
            "lat": lat * 1e-7 if lat else None,
            "lon": lon * 1e-7 if lon else None,
            "alt_baro_m": alt_baro * 0.5 - 1000 if alt_baro else None,
            "alt_geo_m": alt_geo * 0.5 - 1000 if alt_geo else None,
            "height_m": height * 0.5 - 1000 if height else None,
            "h_acc": acc & 0x0F, "v_acc": acc >> 4,
            "rid_ts_tenths": ts,
            "rid_ts_unix": _rid_time_to_unix(ts, rx_unix),
        })
    elif mtype == 0x4:   # System (operator location)
        lat, lon = struct.unpack("<ii", m[2:10])
        out["operator_lat"] = lat * 1e-7 if lat else None
        out["operator_lon"] = lon * 1e-7 if lon else None
    return out


def decode_pack_or_single(payload: bytes, rx_unix: float) -> list[dict]:
    """Payload starts at an ODID message header; either a single message or a pack (0xF).

    A pack is rejected if its message size is not 25 bytes, if it claims more than 10
    messages, or if the payload is shorter than the messages it claims."""
    if not payload:
        return []
    if payload[0] >> 4 == 0xF:            # message pack: header, size, count, messages
        if len(payload) < 3:
            return []
        size, n = payload[1], payload[2]
        if size != ODID_MSG_SIZE or not 1 <= n <= ODID_PACK_MAX or len(payload) < 3 + n * size:
            return []
        msgs = [decode_message(payload[3 + i * size:3 + (i + 1) * size], rx_unix) for i in range(n)]
        return [m for m in msgs if m]
    m = decode_message(payload[:ODID_MSG_SIZE], rx_unix)
    return [m] if m else []


def parse_beacon_vendor_ie(info: bytes, rx_unix: float):
    """Body of a vendor-specific IE (ID 221) in a beacon. Returns (counter, messages) or None."""
    if len(info) > 5 and info[:3] == ASTM_OUI and info[3] == ASTM_OUI_TYPE:
        return info[4], decode_pack_or_single(bytes(info[5:]), rx_unix)
    return None


def parse_ble_service_data(data, rx_unix: float):
    """Service data for UUID 0xFFFA: application code 0x0D, counter, then one message.
    Returns (counter, messages) or None."""
    if not data or len(data) < 2 + ODID_MSG_SIZE or data[0] != BLE_APP_CODE:
        return None
    msgs = decode_pack_or_single(bytes(data[2:]), rx_unix)
    return (data[1], msgs) if msgs else None


def parse_nan_action(body: bytes, rx_unix: float):
    """Body of a NAN service discovery action frame (after the 802.11 header).
    Returns (counter, messages) or None."""
    i = body.find(NAN_OUI + b"\x13")
    if i < 0:
        return None
    # first ODID message-pack header after the NAN header: 0xF<ver>, size 25
    for j in range(i + 4, len(body) - 3):
        if body[j] >> 4 == 0xF and body[j + 1] == ODID_MSG_SIZE:
            msgs = decode_pack_or_single(body[j:], rx_unix)
            if msgs:
                return body[j - 1], msgs
    return None


# --------------------------------------------------------------------------- logging
class Logger:
    def __init__(self, node: str, outdir: Path):
        outdir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        self.path = outdir / f"rid_{node}_{stamp}.csv"
        self.f = open(self.path, "w", newline="")
        self.w = csv.DictWriter(self.f, fieldnames=FIELDS, extrasaction="ignore")
        self.w.writeheader()
        self.node = node
        self.lock = threading.Lock()
        self.last_id: dict[str, str] = {}   # src -> UAS ID (Basic ID arrives separately)
        self.n = 0

    def write(self, bearer, src, rssi, channel, counter, msgs, rx_unix):
        with self.lock:
            for m in msgs:
                if m.get("uas_id"):
                    self.last_id[src] = m["uas_id"]
            for m in msgs:
                row = {"node": self.node, "rx_time_unix": f"{rx_unix:.6f}",
                       "rx_time_utc": datetime.fromtimestamp(rx_unix, timezone.utc).isoformat(),
                       "bearer": bearer, "src_mac": src, "rssi_dbm": rssi, "channel": channel,
                       "msg_counter": counter, "uas_id": m.get("uas_id") or self.last_id.get(src, "")}
                row.update({k: v for k, v in m.items() if k != "uas_id"})
                self.w.writerow(row)
                self.n += 1
            self.f.flush()


# --------------------------------------------------------------------------- Wi-Fi
def set_monitor(iface: str, channel: int):
    cmds = [["ip", "link", "set", iface, "down"], ["iw", iface, "set", "monitor", "none"],
            ["ip", "link", "set", iface, "up"], ["iw", iface, "set", "channel", str(channel)]]
    for c in cmds:
        subprocess.run(c, check=True)


def wifi_loop(iface: str, channel: int, log: Logger):
    from scapy.all import Dot11, Dot11Beacon, Dot11Elt, RadioTap, sniff  # noqa: F401

    def handle(pkt):
        rx = time.time()
        if not pkt.haslayer(Dot11):
            return
        rssi = getattr(pkt, "dBm_AntSignal", None)
        src = pkt[Dot11].addr2 or ""
        if pkt.haslayer(Dot11Beacon):
            el = pkt.getlayer(Dot11Elt)
            while el is not None:
                if el.ID == 221:
                    res = parse_beacon_vendor_ie(bytes(el.info), rx)
                    if res and res[1]:
                        log.write("wifi_beacon", src, rssi, channel, res[0], res[1], rx)
                el = el.payload.getlayer(Dot11Elt)
        else:
            # Wi-Fi NAN service discovery frames (action frames) carrying an ODID pack
            res = parse_nan_action(bytes(pkt[Dot11].payload), rx)
            if res:
                log.write("wifi_nan", src, rssi, channel, res[0], res[1], rx)

    sniff(iface=iface, prn=handle, store=False)


# --------------------------------------------------------------------------- Bluetooth
async def ble_loop(log: Logger):
    from bleak import BleakScanner

    uuid = f"0000{BLE_UUID16:04x}-0000-1000-8000-00805f9b34fb"

    def cb(device, adv):
        rx = time.time()
        res = parse_ble_service_data(adv.service_data.get(uuid), rx)
        if res:
            log.write("ble_legacy", device.address, adv.rssi, None, res[0], res[1], rx)

    scanner = BleakScanner(cb, scanning_mode="active")
    await scanner.start()
    while True:
        await asyncio.sleep(1)


# --------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--node", required=True, help="node name, e.g. N1")
    ap.add_argument("--iface", default="wlan1", help="monitor-capable Wi-Fi interface")
    ap.add_argument("--channel", type=int, default=6, help="2.4 GHz channel (Remote ID: 6)")
    ap.add_argument("--out", default="rid_logs")
    ap.add_argument("--no-wifi", action="store_true")
    ap.add_argument("--no-ble", action="store_true")
    ap.add_argument("--skip-monitor-setup", action="store_true")
    a = ap.parse_args()
    if os.geteuid() != 0 and not a.no_wifi:
        sys.exit("Wi-Fi capture needs root: run with sudo")
    log = Logger(a.node, Path(a.out))
    print(f"[{a.node}] logging to {log.path}")
    if not a.no_wifi:
        if not a.skip_monitor_setup:
            set_monitor(a.iface, a.channel)
        threading.Thread(target=wifi_loop, args=(a.iface, a.channel, log), daemon=True).start()
    try:
        if not a.no_ble:
            asyncio.run(ble_loop(log))
        else:
            while True:
                time.sleep(5)
                print(f"[{a.node}] {log.n} messages")
    except KeyboardInterrupt:
        print(f"[{a.node}] stopped, {log.n} messages in {log.path}")


if __name__ == "__main__":
    main()
