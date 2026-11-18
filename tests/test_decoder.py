"""Remote ID decoder tests against reference packets from the Open Drone ID library.

tests/odid_vectors.json was produced by tests/make_odid_vectors.c, compiled against
opendroneid-core-c (https://github.com/opendroneid/opendroneid-core-c, commit a92bd5f).
Each case holds encoded Basic ID, Location and System messages, a message pack, a complete
Wi-Fi beacon frame and a complete Wi-Fi NAN action frame built by that library, and the
values the library itself decodes. Run with:  python3 -m pytest tests
"""
import json
import math
import struct
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "field"))
import rid_receiver as rr  # noqa: E402

CASES = json.loads((Path(__file__).parent / "odid_vectors.json").read_text())
RX = 1_800_000_000.0 + 1234.0        # a receive time inside some UTC hour
HDR = 24                             # IEEE 802.11 management header length


def _close(a, b, tol):
    return a is not None and b is not None and math.isclose(a, b, abs_tol=tol)


def _check_location(m, e):
    assert m["msg_type"] == 1
    assert m["status"] == e["status"]
    assert _close(m["lat"], e["lat"], 1e-7) and _close(m["lon"], e["lon"], 1e-7)
    assert _close(m["direction_deg"], e["direction_deg"], 1e-6)
    assert _close(m["speed_h_ms"], e["speed_h_ms"], 1e-6)
    assert _close(m["speed_v_ms"], e["speed_v_ms"], 1e-6)
    for k in ("alt_baro_m", "alt_geo_m", "height_m"):     # -1000 m means "unknown"
        assert (m[k] is None) if e[k] == -1000 else _close(m[k], e[k], 1e-6), k
    assert m["h_acc"] == e["h_acc"] and m["v_acc"] == e["v_acc"]
    assert m["rid_ts_tenths"] == e["rid_ts_tenths"]


def test_single_messages():
    for c in CASES:
        e = c["expected"]
        loc = rr.decode_message(bytes.fromhex(c["location_msg"]), RX)
        _check_location(loc, e)
        bid = rr.decode_message(bytes.fromhex(c["basic_id_msg"]), RX)
        assert bid["uas_id"] == c["uas_id"]
        assert bid["id_type"] == e["id_type"] and bid["ua_type"] == e["ua_type"]
        sysm = rr.decode_message(bytes.fromhex(c["system_msg"]), RX)
        if e["operator_lat"] == 0:
            assert sysm["operator_lat"] is None
        else:
            assert _close(sysm["operator_lat"], e["operator_lat"], 1e-7)
            assert _close(sysm["operator_lon"], e["operator_lon"], 1e-7)


def test_message_pack():
    for c in CASES:
        msgs = rr.decode_pack_or_single(bytes.fromhex(c["pack"]), RX)
        types = [m["msg_type"] for m in msgs]
        assert 0 in types and 1 in types and 4 in types
        _check_location(next(m for m in msgs if m["msg_type"] == 1), c["expected"])


def test_wifi_beacon_frame():
    for c in CASES:
        frame = bytes.fromhex(c["beacon_frame"])
        # walk the information elements after the fixed beacon fields (8 + 2 + 2 bytes)
        i, found = HDR + 12, None
        while i + 2 <= len(frame):
            eid, ln = frame[i], frame[i + 1]
            if eid == 221:
                found = rr.parse_beacon_vendor_ie(frame[i + 2:i + 2 + ln], RX)
                if found:
                    break
            i += 2 + ln
        assert found, "ASTM vendor IE not found"
        counter, msgs = found
        assert counter == c["beacon_counter"]
        _check_location(next(m for m in msgs if m["msg_type"] == 1), c["expected"])


def test_wifi_nan_frame():
    for c in CASES:
        res = rr.parse_nan_action(bytes.fromhex(c["nan_frame"])[HDR:], RX)
        assert res, "no ODID pack found in NAN frame"
        counter, msgs = res
        assert counter == c["nan_counter"]
        _check_location(next(m for m in msgs if m["msg_type"] == 1), c["expected"])


def test_bluetooth_legacy_service_data():
    for c in CASES:
        data = bytes([0x0D, 7]) + bytes.fromhex(c["location_msg"])
        counter, msgs = rr.parse_ble_service_data(data, RX)
        assert counter == 7
        _check_location(msgs[0], c["expected"])


def test_malformed_input_is_rejected():
    good = bytes.fromhex(CASES[0]["pack"])
    assert rr.decode_pack_or_single(b"", RX) == []
    assert rr.decode_pack_or_single(good[:2], RX) == []                 # header only
    assert rr.decode_pack_or_single(good[:3 + 25 * 1], RX) == []        # truncated pack
    bad_size = bytes([good[0], 24]) + good[2:]
    assert rr.decode_pack_or_single(bad_size, RX) == []
    too_many = bytes([good[0], 25, 11]) + good[3:]
    assert rr.decode_pack_or_single(too_many, RX) == []
    assert rr.decode_message(bytes(10), RX) == {}                       # short message
    assert rr.parse_beacon_vendor_ie(b"\x00\x11\x22\x0d\x01" + good, RX) is None   # wrong OUI
    assert rr.parse_nan_action(b"\x04\x09\x50\x6f\x9a\x13" + bytes(40), RX) is None
    assert rr.parse_ble_service_data(bytes([0x0C, 1]) + bytes(25), RX) is None


def test_timestamp_conversion_across_the_hour():
    hour0 = 1_800_000_000.0 - (1_800_000_000.0 % 3600)
    # broadcast at hh:59:59.9, received 0.3 s later in the next hour
    assert math.isclose(rr._rid_time_to_unix(35999, hour0 + 3600 + 0.2), hour0 + 3599.9)
    # broadcast at hh:00:00.1, receiver clock still 0.3 s before the hour
    assert math.isclose(rr._rid_time_to_unix(1, hour0 + 3599.8), hour0 + 3600.1)
    # ordinary case inside the hour
    assert math.isclose(rr._rid_time_to_unix(12345, hour0 + 1235.0), hour0 + 1234.5)
    assert rr._rid_time_to_unix(0xFFFF, hour0) is None
    assert rr._rid_time_to_unix(36001, hour0) is None


def test_location_unknown_values():
    m = bytearray(25)
    m[0] = 0x12                     # Location, protocol version 2
    m[1] = 0x20                     # status 2, E/W 0, multiplier 0
    m[2] = 181                      # direction 181 degrees (E/W flag 0)
    m[3] = 255                      # speed unknown
    m[4] = 126                      # vertical speed unknown (63 m/s)
    struct.pack_into("<H", m, 21, 0xFFFF)
    d = rr.decode_message(bytes(m), RX)
    assert d["speed_h_ms"] is None and d["speed_v_ms"] is None
    assert d["lat"] is None and d["lon"] is None
    assert d["rid_ts_unix"] is None
