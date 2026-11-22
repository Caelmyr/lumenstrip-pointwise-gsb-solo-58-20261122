# LumenStrip

LumenStrip is a local workbench for camera photographs of a visible-light spectrum. It connects a drawn detector strip and a user-supplied wavelength ruler to measured light, filter comparisons and immutable evidence. It accepts ordinary single-frame PNG, JPEG, TIFF and WebP. Synthetic examples are identified on screen.

## Start

```sh
docker compose up --build --wait
```

Open http://127.0.0.1:19964. Set `LUMENSTRIP_PORT` and pass a different Compose project name for a separate instance. The named volume keeps studio records, original images and reading files after the web container is recreated. For a local Python 3.11 installation, install the pins from `requirements.txt` and run `./start.sh`.

## Read a spectrum

1. From the home screen open **Open synthetic demo**, or create a studio and upload a camera observation. A studio keeps one instrument/session. Set an exposure time and mark the role of each capture if comparing multiple images.
2. Open **Strip**, mark the left end, several points along the same spectral band, and the right end. With two or more marks the strip follows your points as a curved path: each column is read only from rows inside the search band around your line, so a nearby brighter reflection is never substituted. Where two lines are too close to separate, that span is left unread rather than guessed. Preview the selected detector pixels and the old straight-line reading side by side, then save the geometry. Choose **Straight two-point line** at any time to switch back to the previous reading.
3. Under **Wavelength**, enter at least two known wavelengths and click their actual detector columns. Choose linear or line-by-line interpolation and set whether the fit extends outside the anchors. Preview and save the ruler. Ruler CSV import uses relative columns, native pixel columns, or both.
4. Open **Inspect**, select **Strip spectrum**, and run the reading. Click the plotted signal to see its native camera pixels and the actual rows contributing. **Rows taken on original** marks the untouched original with the exact native rows each column used, every span left unread, and the old straight-line path; retain it to keep that evidence. Other tools use the same versioned input.
5. For **Filter transmission**, tick sample, open reference and optional dark reference in that order. **Aligned repeats** keeps each selected spectrum independent.
6. Choose **Retain reading** to keep the tool settings, source versions, chart and CSV. The retained reading is still tied to the original bytes when geometry or fit versions change. Download a portable evidence packet or reopen an earlier measurement from **Retained**. **Check evidence ZIP** checks the manifest, actual member bytes and declared original hashes.

## Normal capabilities

- Inspect camera support with original code distributions, a spatial grid or native patch.
- Draw, preview and retain a strip with background sidebands; review lost contiguous portions and cross-strip geometry. A guided strip follows user-marked points on one band, keeps to the same band near a brighter neighbour, leaves ambiguous spans unread, and reports the straight-line alternative.
- Fit or import a wavelength ruler and verify native anchors and mapped range.
- Read sampled spectra, source-coordinate evidence, detector channels and channel clipping.
- Measure emission peaks and observed widths, broad band shapes, point/line energy, signal distributions and supported integration.
- Compare detector exposure response, repeat spread, shifts and common wavelength support.
- Analyze open reference and dark-corrected filter transmission, density and requested pass/stop bands independently.
- Inspect detector spots and background asymmetry, and compare simultaneously captured parallel strips.
- Generate and import a clearly labelled synthetic detector scene.
- Freeze readings, inspect saved camera originals, compare source versions, export a printable sheet and check a downloaded ZIP.

## Boundaries

Images are capped at 24 MB and 6.5 million decoded pixels apiece; a four-input reading is capped at 15 million pixels total. A studio stores up to 80 observations and 200 retained measurements. A guard rejects simultaneous heavy readings beyond the local worker budget. Native supports require essentially opaque pixels. Background subtraction uses the configured mean or median over measured sideband rows. For ordinary encoded RGB, detector codes are interpreted as sRGB, with usable embedded profiles converted; subsequent intensity comparisons use linearized camera values. This is a relative camera workbench rather than absolute spectroradiometry or a human colour matching model. Axis models depend on measured anchors. Interpolation intentionally leaves unsupported intervals unmeasured. A guided strip is a reading aid, not a claim about where a line "really" is: it hides any span it cannot separate from a second bright line, and it must be checked against the marked rows. The exported archives contain camera originals only when that option is chosen.

The synthetic lamp and gel images are illustrations with known geometry, not laboratory calibration records. The normal product intentionally keeps input observations separate in the repeat, exposure and filtering views.
