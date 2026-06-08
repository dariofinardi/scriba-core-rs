# scriba-core

**Headless speech-to-text engine for Rust** — audio decoding, resampling, Whisper inference, microphone capture, and speaker diarization, with no UI dependencies.

![License](https://img.shields.io/badge/license-AGPL--3.0--or--later-blue)
![Platform](https://img.shields.io/badge/platform-Windows-blue)

## Why scriba-core

Cloud transcription services come with real limitations: network latency, usage-based costs, and privacy risks for voice data. `scriba-core` brings transcription entirely on-device, using [whisper.cpp](https://github.com/ggerganov/whisper.cpp) — the C/C++ implementation of OpenAI's [Whisper](https://github.com/openai/whisper) — through native Rust bindings.

The crate is designed as a reusable foundation: it powers [Scriba](https://github.com/dariofinardi/scriba-rs) (the desktop app with UI), but can be integrated into any Rust application that needs speech-to-text without depending on a UI framework.

## Workspace structure

```
scriba-core-rs/
├── core/          # scriba-core library crate
├── whisper-cli/   # CLI transcription tool
└── cmake/         # Build helpers for ARM64 cross-compilation
```

## Features

### Audio decoding
Decode WAV, MP3, and OGG files to mono f32 PCM via [Symphonia](https://crates.io/crates/symphonia). Multi-channel audio is automatically mixed to mono.

### Resampling
High-quality sinc resampling to 16 kHz (the sample rate expected by Whisper) via [Rubato](https://crates.io/crates/rubato), using a 256-tap BlackmanHarris2 windowed filter. Passthrough when the source is already at 16 kHz.

### Transcription
whisper.cpp inference through the `TranscriberBackend` trait, returning `Segment`s with start/end timestamps. Supports:
- Automatic language detection or explicit language selection
- Translation to English
- Configurable thread count

### Live microphone capture
Real-time transcription from the system microphone via [cpal](https://crates.io/crates/cpal). Audio is captured in overlapping windows to avoid cutting words at chunk boundaries.

### Speaker diarization (optional)
Identify *who* is speaking in each segment via [sherpa-rs](https://crates.io/crates/sherpa-rs). Enable with the `diarize` feature flag. Segmentation and embedding models are downloaded automatically on first use from the [sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx) project.

## Feature flags

| Flag | Description |
|------|-------------|
| `diarize` | Speaker diarization via sherpa-rs + ONNX Runtime |
| `cuda` | NVIDIA GPU acceleration (CUDA) |
| `metal` | Apple GPU acceleration (Metal) |
| `vulkan` | Cross-platform GPU acceleration (Vulkan) |

## Usage as a library

Add to your `Cargo.toml`:

```toml
[dependencies]
scriba-core = { git = "https://github.com/dariofinardi/scriba-core-rs.git", tag = "v20260606" }
```

### Transcribe a file

```rust
use scriba_core::audio::{decode_audio_file, resample_to_16khz};
use scriba_core::transcriber::{TranscribeParams, TranscriberBackend, WhisperCppBackend};

let decoded = decode_audio_file("recording.wav".as_ref())?;
let pcm = resample_to_16khz(&decoded.samples, decoded.sample_rate)?;

let mut backend = WhisperCppBackend::new("ggml-large-v3-turbo.bin".as_ref())?;
let segments = backend.transcribe(&pcm, &TranscribeParams::default())?;

for seg in &segments {
    println!("[{} → {}] {}", seg.start_ms, seg.end_ms, seg.text.trim());
}
```

### Live microphone transcription

```rust
use scriba_core::mic;
use scriba_core::transcriber::{TranscribeParams, WhisperCppBackend};

let mut backend = WhisperCppBackend::new("model.bin".as_ref())?;
mic::run_live(&mut backend, &TranscribeParams::default())?;
```

## whisper-cli

A ready-to-use command-line tool included in the workspace.

### File transcription

```sh
cargo run -p whisper-cli --release -- --model ggml-large-v3-turbo.bin file recording.wav
```

### Live microphone

```sh
cargo run -p whisper-cli --release -- --model ggml-large-v3-turbo.bin listen
```

### With speaker diarization

```sh
cargo run -p whisper-cli --release --features diarize -- --model ggml-large-v3-turbo.bin --diarize file meeting.wav
```

### Options

```
Usage: whisper-cli [OPTIONS] --model <MODEL> <COMMAND>

Commands:
  file    Transcribe an audio file (wav, mp3, ogg)
  listen  Live transcription from the system microphone

Options:
  -m, --model <MODEL>        Path to the GGML/GGUF model file [env: WHISPER_MODEL]
  -l, --language <LANGUAGE>  Language code ("it", "en") or "auto" [default: auto]
  -t, --translate            Translate to English instead of transcribing
  -d, --diarize              Enable speaker diarization (requires --features diarize)
  -j, --threads <THREADS>    Number of inference threads [default: all cores]
```

## Whisper models

GGML models can be downloaded from [Hugging Face](https://huggingface.co/ggerganov/whisper.cpp/tree/main). Scriba uses Q5 quantized versions to reduce size while maintaining quality close to float16:

| Model | Size | Speed | Quality | Recommended use |
|-------|------|-------|---------|-----------------|
| `ggml-small-q5_1.bin` | 190 MB | ★★★★ | ★★ | Quick notes, drafts |
| `ggml-medium-q5_0.bin` | 515 MB | ★★★ | ★★★ | Good balance |
| `ggml-large-v3-turbo-q5_0.bin` | 574 MB | ★★★ | ★★★★ | Maximum accuracy |

All models support 90+ languages.

## Building

### Standard build

```sh
cargo build --release
```

### With diarization

```sh
cargo build --release --features diarize
```

### Windows ARM64 (Qualcomm Snapdragon)

Requires Ninja and clang-cl. Set the following environment variables:

```powershell
$env:PATH = "cmake;" + $env:PATH
$env:CMAKE_TOOLCHAIN_FILE = "cmake/arm64-toolchain.cmake"
$env:CMAKE_GENERATOR = "Ninja"
$env:CMAKE_C_COMPILER = "clang-cl"
$env:CMAKE_CXX_COMPILER = "clang-cl"
$env:CMAKE_ASM_COMPILER = "clang-cl"
```

Then build normally with `cargo build --release`.

## Key dependencies

| Crate | Role |
|-------|------|
| [whisper-rs](https://github.com/dariofinardi/whisper-rs) | Rust bindings for whisper.cpp (fork with Windows fixes) |
| [symphonia](https://crates.io/crates/symphonia) | Multi-format audio decoding |
| [rubato](https://crates.io/crates/rubato) | High-quality sinc resampling |
| [cpal](https://crates.io/crates/cpal) | Cross-platform audio capture |
| [sherpa-rs](https://crates.io/crates/sherpa-rs) | Speaker diarization (optional) |

## Platforms

| Platform | Status |
|----------|--------|
| Windows x86_64 | Tested |
| Windows ARM64 (Qualcomm Snapdragon X Elite) | Tested |
| macOS (Apple Silicon / Intel) | Not yet tested |
| Linux x86_64 | Not yet tested |

The codebase is cross-platform by design (whisper.cpp, Symphonia, and cpal all compile on every platform), but only Windows has been verified so far.

## License

AGPL-3.0-or-later — see [LICENSE](LICENSE).

Copyright © 2026 Dario Finardi
