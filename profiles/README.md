# Product profiles

One directory per uRAD product with everything the out-of-box demo needs:
a `config_radar.json` (serial ports, output options, GUI plot ranges) and
the chirp configuration files exported from the TI mmWave Demo Visualizer
(SDK 3.06).

| Profile | Product | Chip | Band |
|---|---|---|---|
| [`automotive/`](automotive) | uRAD Automotive | AWR1843AoP | 77 GHz |
| [`automotive-hpa/`](automotive-hpa) | uRAD Automotive HPA | AWR1843 ISK | 77 GHz |
| [`industrial/`](industrial) | uRAD Industrial | IWR6843AoP | 60 GHz |

## Usage

Edit the serial ports in the profile's `config_radar.json` (or override them
on the command line) and run:

```bash
urad-mmwave --config profiles/industrial/config_radar.json --data-port COM7 --control-port COM8
```

Add `--gui` for the live point cloud viewer. Relative paths inside the JSON
are resolved against the profile directory, so the command works from
anywhere. Output files are written relative to your current directory.

## Temperature monitoring

Each profile also ships `chirp_config_temperature.cfg`, identical to the
standard chirp configuration but with the temperature TLV enabled. To use it:

```bash
urad-mmwave --config profiles/industrial/config_radar.json --chirp profiles/industrial/chirp_config_temperature.cfg
```

and set `"save_temperature": true` / `"print_temperature": true` in the
`output`/`display` sections if you want the readings saved or printed.

> The `.cfg` files are generated with the TI mmWave Demo Visualizer — do not
> edit them by hand. To create a custom radar configuration, use the
> Visualizer and save the result as a new `.cfg` file.
