# DCP-o-matic — GPU JPEG2000 fork

DCP-o-matic is a program to generate DCPs (digital cinema packages).
Please see [dcpomatic.com](https://dcpomatic.com/) for the upstream project.

**This fork adds a GPU-accelerated JPEG2000 encode path** (Slang/Vulkan, via an
external frame server) plus cinema audio automation, a GUI to drive it, and
HTJ2K conformance fixes. DCP-o-matic keeps full ownership of demux, colour, MXF
wrapping and the DCP package; only the per-frame `XYZ → .j2c` step is offloaded
to the GPU encoder over a Unix socket.

The GPU encoder itself lives in a separate repository
([jpeg2000-gpu-encoder](https://github.com/amitbar05/jpeg2000-gpu-encoder)); the
canonical copies of every file this fork adds/changes, the
`slang_integration.patch`, and the full technical write-up live there under
`encoder/integration/dcpomatic/`. Everything here is gated behind
`-DDCPOMATIC_SLANG` (and the runtime `DCPOMATIC_SLANG` env var / Preferences
switch), so an unconfigured build is stock DCP-o-matic.

## Features added to DCP-o-matic

### GPU JPEG2000 encoding
- **GPU encode path** — a `SlangJ2KEncoderThread` runs beside the existing CPU
  (OpenJPEG) and Grok threads. Each frame's pixels go to the frame server; the
  `.j2c` codestream comes back for libdcp's MXF writer. DCI-conformant output.
- **Two Tier-1 coders** — **HT** (HTJ2K / JPEG 2000 Part 15, the fast default,
  ~3× on the GPU, fills the DCI budget) and **MQ** (highest PSNR, widest decoder
  compatibility). Chosen in Preferences, the export-time coder-picker dialog, or
  per-connection on the wire (`J2KO`).
- **Coder enforced & verified per frame** — the requested coder/bit-rate are
  re-sent on every reconnect, structural refusals fail the job, and every
  returned frame is ground-truth checked (Rsiz HT bit + byte cap) so a stale or
  misconfigured server can't silently produce the wrong coder. An
  all-encoder-threads-dead export now fails with the stored error instead of
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
- **Smart-centre upmix → 5.1** — a mono/stereo source is upmixed to a 5.1-shaped
  mix (L, R, C, LFE, Ls, Rs) with dialogue on the **centre** speaker via a
  mid/side **extraction** matrix (`C=(L+R)/2, L'=L−mid, R'=R−mid`; mono → C only).
  Dialogue is removed from L/R rather than doubled as a phantom, LFE and surrounds
  stay silent, and the export raises the film to ≥6 channels so the centre always
  has a slot. Selectable as an AudioProcessor in the DCP audio panel too.
- **GPU auto-gain, on import** — measures the mix peak on the GPU and normalises
  it to just under −3.5 dBFS. Runs **as soon as content is imported** (not only at
  export) and applies an **idempotent, absolute** gain correction, so running it
  on import and again at export never drifts. Result reported inline in the Jobs
  panel.
- **GPU audio stats** — per-channel peak/RMS reduction on the GPU (exact peak),
  batched for efficiency, with a NumPy fallback and kill switch.
- **DCP sound layer** — 24-bit/48 kHz PCM wrapped into a SMPTE sound MXF, added to
  the CPL as MainSound and validated.

### GUI
- **Preferences → GPU (Slang)** — enable the GPU export, pick the coder, set the
  frame-server socket, and toggle the audio + bit-rate automation.
- **Jobs → Make DCP using GPU (Ctrl-Shift-M)** — one-click GPU export that applies
  bit-rate matching, the smart-centre 5.1 upmix, the audio auto-gain, and the
  coder picker, then makes the DCP.
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
- **Independent DCI conformance gate** — the frame server can validate every
  frame against the full cinema profile and fail fast on any non-conformant one.

## Build

Apply `encoder/integration/dcpomatic/slang_integration.patch`, drop in the added
`src/lib/*` and `src/wx/*` files from that directory, and build with
`-DDCPOMATIC_SLANG`. See that directory's `README.md` for the full build wiring,
runtime instructions, environment-variable reference, and protocol details.
