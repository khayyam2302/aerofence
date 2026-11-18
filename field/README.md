# AeroFence field test: Raspberry Pi receivers and a Remote ID drone

This folder holds everything for the field validation requested by Reviewer 1: the receiver
software for each node, the analysis script, and the test protocol.

| File | Purpose |
|---|---|
| `rid_receiver.py` | Runs on each Raspberry Pi. Captures Remote ID over Wi-Fi (beacon and NAN) and Bluetooth, decodes it, and writes one CSV row per message with the node's own timestamp |
| `field_analysis.py` | Runs on your laptop after the test. Computes time to first fix, latency, packet reception, position error and a path-loss fit, runs the AeroFence fusion on the real messages, and draws the figures |
| `make_test_dataset.py` | Pipeline test only. Writes a synthetic folder so you can check `field_analysis.py` before flying. Never report its output |

The decoder has been checked against the official Open Drone ID C library: messages encoded by `opendroneid-core-c` decode to the same values.

---

## 1. Hardware per node (2 to 4 nodes)

| Item | Example | Note |
|---|---|---|
| Raspberry Pi 4 (2 GB+) or Pi 5 | with 32 GB microSD and case | |
| USB Wi-Fi adapter with monitor mode | MediaTek MT7612U (e.g. Alfa AWUS036ACM) or MT7921AU | The built-in Pi Wi-Fi cannot capture Remote ID frames reliably |
| USB GPS receiver | u-blox 7/8 (e.g. VK-162) | Gives each node an accurate clock, needed for latency |
| Power bank | USB-C PD, 20,000 mAh | About 6 to 8 hours per node |
| Stand | tripod or box, about 1.5 m high | Same height for every node |

The Pi's built-in Bluetooth receives Bluetooth legacy advertising. Many drones, including DJI models, send Remote ID over Wi-Fi (beacon or NAN), so the Wi-Fi adapter is the important part. Check the bearer of your test drone with the OpenDroneID OSM app.

## 2. Software setup (once per node, about 20 minutes)

Install Raspberry Pi OS (64-bit, Bookworm). Then in a terminal:

```bash
sudo apt update
sudo apt install -y python3-pip python3-scapy python3-pandas iw gpsd gpsd-clients chrony
pip3 install --break-system-packages bleak
mkdir -p ~/aerofence && cd ~/aerofence      # copy rid_receiver.py here
```

**Time synchronisation with GPS** (so every node agrees on the time to a few ms):

```bash
sudo sed -i 's|^DEVICES=.*|DEVICES="/dev/ttyACM0"|; s|^GPSD_OPTIONS=.*|GPSD_OPTIONS="-n"|' /etc/default/gpsd
echo "refclock SHM 0 refid GPS precision 1e-1 offset 0.1 delay 0.2" | sudo tee -a /etc/chrony/chrony.conf
sudo systemctl restart gpsd chrony
chronyc sources -v        # after 2-5 minutes outdoors, GPS should show with '*' or '+'
```

If you also have internet at the field (phone hotspot), chrony uses NTP too. Write down the
`chronyc tracking` "System time" offset at the start and end of each test.

**Check the Wi-Fi adapter** (usually `wlan1`; `wlan0` is the built-in one):

```bash
iw dev                                   # find the adapter name
sudo python3 rid_receiver.py --node N1 --iface wlan1 --channel 6
```

## 3. Before the flight test

1. **Drone firmware.** Update the test drone to a firmware version that supports Remote ID broadcast, and record the model, firmware version and region setting. Support depends on all three.
2. **Check that it broadcasts.** Power on the drone outdoors with GPS lock, props off. Run a node or the OpenDroneID OSM Android app. You should see the serial number and position. Write the serial number into `test.json` as `uas_id`.
3. If the drone does not broadcast in Saudi Arabia, fit an ESP32-S3 module running ArduRemoteID with its own GPS on top of the drone, and use that module's ID.
4. **Permission.** GACA and university approval for the test area; keep below 120 m and in line of sight.
5. **Choose a virtual fence.** For a campus field, a square of 150 to 200 m side works well. Write its centre (lat, lon) and side into `test.json`.

## 4. Node layout

* 4 nodes: at the four corners of the virtual fence, or 20 to 30 m outside them.
* 2 nodes: on opposite corners.
* Survey each node position with a phone GPS (average 30 s) and write it into `nodes.json`.
* Antennas vertical, all nodes at the same height, nothing metal next to the antenna.

## 5. Flight plan (one battery, about 20 minutes)

| # | Run | What it measures | Repeat |
|---|---|---|---|
| A | Power on on the ground at the edge of the area; take off when all nodes show messages | Time to first fix from power-on | 3 times |
| B | Hover 60 s each at 50, 100, 200, 300 m from the fence centre, 60 m height | Packet reception and RSSI against distance | once |
| C | Straight pass through the fence at 8 to 10 m/s, 60 m height, entering on different sides | Alert latency at the boundary | 5 to 10 passes |
| D | Fly along one fence edge, 10 to 20 m inside, for 60 s | Boundary jitter (flip rate) | 2 times |
| E | Optional: one node switched off during run C | Effect of losing a receiver | 2 passes |

Keep the drone's flight log recording the whole time. Write down the UTC time of each run start and end.

## 6. After the test

Copy the files into one folder per test day:

```
test_2026-12-xx/
  logs/            rid_N1_*.csv, rid_N2_*.csv, ...
  nodes.json
  test.json
  flight.csv       drone flight log as CSV (for DJI drones: Airdata or PhantomHelp Log Viewer export)
```

For DJI drones, the app logs are on the phone or tablet (for DJI GO 4, under `DJI/dji.go.v4/FlightRecord/`); upload the `.txt` log to airdata.com or phantomhelp.com/LogViewer and download the CSV. For other drones, export the log to CSV with time, latitude, longitude and altitude columns, or use the GPS-track format in `gps_track_synthetic_template.csv`.

Then run:

```bash
python3 field_analysis.py test_2026-12-xx
```

It writes `results/summary.json`, `results/per_node.csv` and four figures. Send me the whole folder.

## 6b. Ground drive-through test (no flight needed)

This test answers Reviewer 1's request for TTFF and alert latency from real broadcasts with
2 to 4 receivers, without flying. The transmitter rides in a car (or is carried) across the
virtual zone while a phone logs the true position.

**Transmitter:** the test drone powered on with motors off (if the OpenDroneID OSM app shows its
Remote ID), or an ESP32 Remote ID beacon fed by its own GPS. Put it on the car roof or dashboard.

**Ground truth:** a phone GPS logger at 1 Hz (for example "GPS Logger" on Android), exported to
CSV with these columns (same as `gps_track_synthetic_template.csv`):

    timestamp,latitude,longitude,altitude_m,speed_mps,heading_deg,data_label
    2027-01-15T08:00:00Z,26.2187,50.0525,12.0,10.0,90.0,MEASURED

- `timestamp` must be ISO 8601 in **UTC** (end with Z). Riyadh time is UTC+3.
- `data_label`: write `MEASURED` for real data. Any value containing `SYNTHETIC` makes the
  analysis stamp every output "SYNTHETIC - NOT MEASURED".
- `altitude_m`, `speed_mps`, `heading_deg` are kept for reference; the analysis uses
  `tx_height_m` from test.json as the transmitter height above ground.

**test.json** for a drive test:

    {"fence_center": [lat, lon], "fence_side_m": 200, "warn_side_m": 300,
     "uas_id": "<serial shown in the app>", "flight_log": "gps_track.csv", "tx_height_m": 1.5}

**Drive plan:** 6 to 10 straight passes through the zone at a steady 30 to 40 km/h, with
15 to 30 s of driving outside the warning ring between passes. Then 3 power-on tests at the
zone edge (TTFF) and 60-s stops at 50, 100, 200 and 300 m from a node (reception vs distance).

**Outputs that matter for the paper** (results/summary.json):
- `per_node` and `network`: TTFF, receive delay median/95th percentile, packet reception ratio
- `path_loss`: exponent and shadowing spread, fitted by a truncated-normal maximum-likelihood
  model, because packets below the receiver sensitivity are never logged and an ordinary
  least-squares fit underestimates the exponent (both values are reported)
- `aerofence_offline`: latency of every true zone entry (`entry_latency_each_s`), the median,
  and any entries that raised no alert

**Pipeline check before going out:** `python3 make_test_dataset.py /tmp/dtest --drive` then
`python3 field_analysis.py /tmp/dtest`. These files are synthetic and labelled as such; never
report them.

## 7. Photos for the paper

1. One node close up, parts labelled (Pi, Wi-Fi adapter, GPS, power bank).
2. All nodes together before deployment.
3. The drone (with the ESP32 module, if used).
4. Wide shot of the field showing at least two deployed nodes.
5. The drone in flight over the area, if possible.
6. A screenshot of a node terminal or the app showing the drone being decoded.

Plain background, good light, no faces or ID cards.
