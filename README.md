# DCP-o-matic — GPU JPEG2000 fork

DCP-o-matic is a program to generate DCPs (digital cinema packages).
Please see [dcpomatic.com](https://dcpomatic.com/) for the upstream project.

**This fork adds a GPU-accelerated JPEG2000 encode path** (Slang/Vulkan, via an
external frame server), a one-screen simplified interface, cinema audio
automation, a GUI to drive all of it, and HTJ2K conformance fixes. DCP-o-matic
keeps full ownership of demux, colour, MXF wrapping and the DCP package; only
the per-frame `XYZ → .j2c` step is offloaded to the GPU encoder over a Unix
socket.

The GPU encoder itself lives in a separate repository
([jpeg2000-gpu-encoder](https://github.com/amitbar05/jpeg2000-gpu-encoder)); the
canonical copies of every file this fork *adds*, the `slang_integration.patch`
(which additionally carries every hunk applied to existing upstream files, since
those have no copy), and the full technical write-up live there under
`encoder/integration/dcpomatic/`. Everything here is gated behind
`-DDCPOMATIC_SLANG` (and the runtime `DCPOMATIC_SLANG` env var / Preferences
switch), so an unconfigured build is stock DCP-o-matic.

## Features added to DCP-o-matic

### Simplified interface

`View → Simplified interface` (Shift+Ctrl+S, remembered in
`Config::Slang::simple_ui`) replaces the film editor with a single screen that
does the whole job: add a video, optionally add subtitles, say where the DCP
goes, see what is about to happen to the sound, press one button.

- **One screen, four cards** — video, subtitles, output folder, sound — plus a
  primary *Create DCP* action and an embedded job list for progress. Files can
  be dropped onto the video and subtitle cards or chosen from them; a drop of
  mixed files is routed per extension rather than refused.
- **No menu bar.** The full interface's File/Edit/Jobs/View/Tools/Help strip is
  hidden in this mode — thirty-odd commands do not belong above a screen whose
  promise is three. It is *hidden*, not detached, so the keyboard accelerators
  still work: Shift+Ctrl+S toggles back with no bar to read it off. (Left in
  place on macOS, where the menu bar belongs to the screen rather than the
  window.) The header carries **New…** — the same `file_new()` the menu item
  ran — and **Advanced…**, which restores the full interface and its menu bar.
- **The sound pipeline, drawn** — source channels on the left, the mapping as
  curves into the processing stage, and the DCP channels on the right with a
  meter and a measured level each. Everything shown is read back from the `Film`
  and from the analysis results the job persists on it, never recomputed for the
  display, so the picture cannot drift from what the export actually makes.
- **No Save button** — `metadata.xml` is written whenever the content changes,
  so a project built here is not lost by closing the window.
- **The export asks nothing** — it runs the same flow as *Jobs → Make DCP using
  GPU* with the configured coder instead of showing the coder dialog.
- Both interfaces are siblings in one frame driving one `Film`, so a project
  started in either opens unchanged in the other. This is a different front end,
  not a parallel pipeline.

### GPU JPEG2000 encoding

- **GPU encode path** — a `SlangJ2KEncoderThread` runs beside the existing CPU
  (OpenJPEG) and Grok threads. Each frame's pixels go to the frame server; the
  `.j2c` codestream comes back for libdcp's MXF writer. DCI-conformant output.
- **Two Tier-1 coders** — **MQ** (the default: highest PSNR, widest decoder
  compatibility, and the only one a cinema will play) and **HT** (HTJ2K, ~3× on
  the GPU and fills the DCI budget, but it is JPEG 2000 **Part 15** — SMPTE
  ST 429-4 has no HTJ2K provision, deployed cinema servers do not decode it, and
  a real export was rejected by a third-party verifier, so it is a
  fast-preview/benchmark coder, not a delivery one). Chosen in Preferences, the
  export-time coder-picker dialog, or per-connection on the wire (`J2KO`).
- **Coder enforced & verified per frame** — the requested coder/bit-rate are
  re-sent on every reconnect, structural refusals fail the job, and every
  returned frame is ground-truth checked (Rsiz HT bit + byte cap) so a stale or
  misconfigured server cannot silently produce the wrong coder. An
  all-encoder-threads-dead export fails with the stored error instead of
  deadlocking.
- **Source bit-rate matching** — probes each source video's real bit rate, scales
  it by the codec's J2K-equivalence factor, and sets the DCP's JPEG2000 bandwidth
  to match (floored/capped/rounded). Runs **automatically on every content
  import** and again at export.
- **Efficient transport** — colour tables sent once then RGB48 frames with a
  bit-exact GPU `convert_to_xyz` (`J2KC`/`J2KG`), a `/dev/shm` zero-copy frame
  path (`J2KS`/`J2KH`), and a classic XYZ payload fallback (`J2KF`).
- **Multi-GPU & heterogeneous** — a comma-separated socket list runs one server
  process per GPU (GIL-dodging, truly parallel); an optional mode also keeps a
  CPU pool draining the queue.

### Cinema audio automation

- **Smart centre → L, C, R** — *"Smart centre (mono/stereo to L, C, R)"*, an
  `AudioProcessor` that builds the centre channel a mono or stereo source does
  not have. It is a fixed matrix, not source separation: what lands in the
  centre is whatever the fronts have in common.
  - **Stereo** is a mid/side **extraction**: `C = (L+R)/2`, `L' = L−C`,
    `R' = R−C`. The centre is *moved* there, not copied there — `L'+C` is
    exactly your original left — so dialogue plays from the centre speaker
    alone rather than the centre plus a phantom in L/R.
  - **Mono** is a centre-dominant **spread**: `C = M`, `L = R = M/2` (6 dB
    down), on its own input leg, since no linear matrix on (L, R) can produce
    both behaviours. Pre-existing projects are moved onto that leg by
    `Film::migrate_smart_center_mono_mapping()`.
  - **Surrounds and the LFE are never invented from front content** and stay
    silent unless the source already has them (ISDCF Doc 4 Note 1: not every
    channel need be present). The export raises the film to ≥6 channels so the
    centre always has a slot.
- **GPU auto-gain, on import** — measures the mix peak on the GPU and normalises
  the loudest channel to just under **−3.5 dBFS**, in both directions: a quiet
  mix is brought up, a hot one is brought down. The boost is uncapped, so the
  target is always reached, with a −60 dBFS floor below which a track is treated
  as silence rather than amplified by 60–140 dB. Runs **as soon as content is
  imported** (not only at export) and applies an **idempotent, absolute**
  correction, so running it on import and again at export never drifts. Result
  reported inline in the Jobs panel.
- **GPU audio stats** — per-channel peak/RMS reduction on the GPU (exact peak),
  batched for efficiency, with a NumPy fallback and kill switch. The job's peak
  is cross-checked against local ground truth rather than trusted.
- **DCP sound layer** — 24-bit/48 kHz PCM wrapped into a SMPTE sound MXF, added
  to the CPL as MainSound, with MCA soundfield/channel labelling and an
  RFC 5646 spoken language, and validated.

### GUI

- **Preferences → GPU (Slang)** — enable the GPU export, pick the coder, set the
  frame-server socket, and toggle the audio + bit-rate automation.
- **Jobs → Make DCP using GPU (Ctrl-Shift-M)** — one-click GPU export that applies
  bit-rate matching, the smart-centre upmix, the audio auto-gain, and the coder
  picker, then makes the DCP.
- **Coder-picker dialog** — an HT-vs-MQ chooser with a plain-language summary of
  each, shown at export time.
- **Directory-chooser "New Folder" fix** — folder pickers/dialogs allow creating
  new folders again on GTK (also submitted upstream as PR #44).

### Conformance

- **libdcp HTJ2K verifier** — the bundled verifier is taught JPEG 2000 Part 15
  (CAP/CPF markers, Rsiz profile, code-block style), so HT DCPs verify with zero
  codestream errors instead of spurious "invalid Rsiz / unknown marker" noise.
- **2K guard-bits fix** — 2K DCI streams emit 1 guard bit (4K keeps 2) per DCI /
  SMPTE Bv2.1; the encoder's own `dci_validate` gate enforces it.
- **CPL `CompositionMetadataAsset`** — written for SMPTE Bv2.1 compliance and
  checked by the package validator.
- **Independent DCI conformance gate** — the frame server can validate every
  frame against the full cinema profile and fail fast on any non-conformant one.
  Exports from this fork have been verified clean by four independent oracles
  (the encoder's own `dci_validate`, asdcplib, libdcp's `dcpverify`, and
  Clairmeta).

## Build

Apply `encoder/integration/dcpomatic/slang_integration.patch`, drop in the added
`src/lib/*` and `src/wx/*` files from that directory, and build with
`-DDCPOMATIC_SLANG`. See that directory's `README.md` for the full build wiring,
runtime instructions, environment-variable reference, and protocol details.
