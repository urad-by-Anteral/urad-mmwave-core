# urad-mmwave

**SDK Python común para los sensores radar mmWave [uRAD](https://urad.es) de Anteral.**

*Read this in [English](README.md).*

Esta librería proporciona los componentes compartidos por todos los productos
uRAD basados en chips mmWave AWR/IWR de Texas Instruments con el firmware
out-of-box demo (mmWave SDK 3.x):

- Comunicación serie con el radar (USB de doble puerto o configuraciones
  single-UART como el adaptador para Raspberry Pi, incluido el reset del chip
  por GPIO).
- Envío de la configuración de chirp exportada desde el TI mmWave Demo
  Visualizer.
- Parser TLV robusto: resincronización por palabra de sincronismo, salto de
  TLVs desconocidos, timeouts acotados y cierre limpio (siempre se envía
  `sensorStop` al salir).
- Salida de nube de puntos y temperatura en el formato de texto clásico de
  uRAD.
- Herramienta de línea de comandos lista para usar: `urad-mmwave`, con visor
  2D de nube de puntos en tiempo real opcional (`--gui`).
- [Perfiles de producto](profiles) listos para usar (puertos serie, rangos
  de plot y configuraciones de chirp) para cada producto soportado.
- Clientes de aplicación construidos sobre el mismo núcleo:
  `urad-level-sensing` (level sensing de alta precisión, AWR e IWR),
  `urad-people-tracking` (3D people tracking — antes "people counting" —
  estándar y overhead), `urad-vital-signs` (constantes vitales con people
  tracking), `urad-area-scanner` (escáner de área con detección de objetos
  estáticos y zonas de seguridad), `urad-automated-doors` (disparo de
  puertas y portones automáticos), `urad-small-obstacle` (detección de
  obstáculos pequeños para robots móviles), `urad-cpd` (ocupación de
  cabina / detección de presencia infantil con clasificación adulto-niño)
  y `urad-mrr` (radar de medio alcance ADAS con clustering, tracking y
  asistencia al aparcamiento). Cada uno requiere su firmware dedicado,
  distribuido en los repositorios de producto.

Productos soportados: **uRAD Automotive** (AWR1843AoP, 77 GHz), **uRAD
Automotive HPA** (AWR1843 ISK, 77 GHz) y **uRAD Industrial** (IWR6843AoP,
60 GHz). Los binarios de firmware se distribuyen como assets de release en
el repositorio de cada producto.

## Instalación

Requiere Python 3.9 o superior.

```bash
pip install git+https://github.com/urad-by-Anteral/urad-mmwave-core.git
```

Para configuraciones single-UART en Raspberry Pi (soporte de reset por GPIO):

```bash
pip install "urad-mmwave[rpi] @ git+https://github.com/urad-by-Anteral/urad-mmwave-core.git"
```

Para el visor de nube de puntos en tiempo real (`--gui`):

```bash
pip install "urad-mmwave[gui] @ git+https://github.com/urad-by-Anteral/urad-mmwave-core.git"
```

## Inicio rápido (línea de comandos)

1. Conecta tu radar uRAD por USB e identifica sus dos puertos serie
   (control ≈ 115200 baudios, datos ≈ 921600 baudios). En Windows, revisa el
   Administrador de dispositivos (`COMx`); en Linux suelen aparecer como
   `/dev/ttyACM0` y `/dev/ttyACM1`.
2. Ejecuta con el [perfil](profiles) de tu producto:

```bash
urad-mmwave --config profiles/industrial/config_radar.json --data-port COM7 --control-port COM8
```

Detén con `Ctrl+C`: el sensor se para y los puertos se liberan
automáticamente. Opciones útiles: `--gui` (visor de nube de puntos en tiempo
real), `--single-port /dev/serial0` (Raspberry Pi), `--duration 10`,
`--max-frames 100`, `--no-save`, `-v`. Las rutas relativas dentro del JSON se
resuelven respecto al directorio del perfil, así que funciona desde cualquier
sitio.

## Inicio rápido (librería)

```python
from urad_mmwave import PointCloudWriter, RadarSession, load_config

config = load_config("config_radar.json")

with RadarSession(config) as session, PointCloudWriter("PointCloud.txt") as out:
    for frame in session.frames():
        # frame.points es un array numpy (N, 6): x, y, z, v, snr, noise
        print(f"Frame {frame.header.frame_number}: {len(frame.points)} puntos")
        out.write(frame.points, frame.timestamp)
```

## Referencia de configuración

| Sección | Clave | Por defecto | Descripción |
|---|---|---|---|
| `control_serial` | `port` | — | UART de control/CLI (p. ej. `COM8`, `/dev/ttyACM0`) |
| | `baudrate` | — | Normalmente `115200` |
| | `timeout` | `0.3` | Timeout de lectura en segundos |
| `data_serial` | `port` | — | UART de datos. Mismo puerto que control = modo single-UART |
| | `baudrate` | — | Normalmente `921600` |
| — | `chirp_config_path` | `./config/chirp_config.cfg` | Configuración de chirp exportada del TI mmWave Demo Visualizer |
| — | `gpio_reset_pin` | `null` | Pin BCM para resetear el chip (solo Raspberry Pi) |
| `packet` | *(todo opcional)* | Valores SDK 3.x | Patrón de sincronismo, formatos de cabecera y límites de sanidad TLV |
| `output` | `save_pointcloud` | `true` | Añade cada frame a `pointcloud_path` |
| | `save_temperature` | `false` | Añade los informes TLV-9 a `temperature_path` |
| `display` | `print_pointcloud` | `true` | Imprime cada punto detectado por stdout |
| | `print_temperature` | `false` | Imprime los informes de temperatura |
| `gui` | `x_range` | `[-10, 10]` | Rango del eje X en metros para el visor `--gui` |
| | `y_range` | `[0, 20]` | Rango del eje Y en metros para el visor `--gui` |

Las rutas relativas (`chirp_config_path`) se resuelven respecto al directorio
del fichero JSON; las rutas de salida son relativas al directorio de trabajo.

### Formato de salida

`PointCloud.txt`: una línea por frame con `x y z v snr noise` repetido por
cada objeto detectado, seguido del timestamp epoch del host.
`Temperature.txt`: los doce campos del TLV-9 seguidos del timestamp.

## Resolución de problemas

- **`SerialException: could not open port`** — nombre de puerto incorrecto,
  puerto en uso por otro programa (p. ej. el TI Visualizer) o falta de
  permisos (en Linux añade tu usuario al grupo `dialout`).
- **No llegan frames** — probablemente los puertos de control y datos están
  intercambiados, o la configuración de chirp no fue aceptada; ejecuta con
  `-v` para ver la respuesta del radar a cada comando.
- **El radar sigue transmitiendo tras un fallo** — no debería ocurrir: la
  sesión siempre envía `sensorStop` al salir. Si mataste el proceso a la
  fuerza, ejecuta el CLI una vez más o reinicia la alimentación del sensor.

## Desarrollo

```bash
git clone https://github.com/urad-by-Anteral/urad-mmwave-core.git
cd urad-mmwave-core
pip install -e .[dev]
pytest          # tests unitarios (sin hardware)
ruff check .    # lint
black .         # formato
```

El parser está cubierto por tests independientes del hardware construidos
sobre paquetes TLV sintéticos — véase [`tests/`](tests/).

## Licencia

[MIT](LICENSE) © Anteral S.L.
