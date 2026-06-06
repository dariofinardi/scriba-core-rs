# scriba-core

Headless speech-to-text engine for Rust, built on [whisper.cpp](https://github.com/ggerganov/whisper.cpp) via [whisper-rs](https://github.com/dariofinardi/whisper-rs). Provides audio decoding, high-quality resampling, transcription, live microphone capture, and optional speaker diarization — all without any UI dependency.

`scriba-core` is the foundation of the [Scriba](https://github.com/dariofinardi/scriba-rs) desktop app, but is designed to be used standalone by any Rust application or CLI tool.

## Workspace structure

```
scriba-core-rs/
├── core/          # scriba-core library crate
├── whisper-cli/   # Command-line transcription tool
└── cmake/         # Build helpers for ARM64 cross-compilation
```

## Features

### Audio decoding
Decode WAV, MP3, and OGG files to mono f32 PCM using [Symphonia](https://crates.io/crates/symphonia). Multi-channel audio is automatically mixed to mono.

### Resampling
High-quality sinc resampling to 16 kHz (the sample rate expected by Whisper) via [Rubato](https://crates.io/crates/rubato), using a 256-tap BlackmanHarris2 windowed filter. Passthrough when the source is already at 16 kHz.

### Transcription
Whisper.cpp inference through a `TranscriberBackend` trait, returning timed `Segment`s with start/end timestamps. Supports:
- Automatic language detection or explicit language selection
- Translation to English
- Configurable thread count

### Live microphone capture
Real-time transcription from the system microphone via [cpal](https://crates.io/crates/cpal). Captures audio in chunked windows with overlap to avoid cutting words at chunk boundaries.

### Speaker diarization (optional)
Identify *who* is speaking via [sherpa-rs](https://crates.io/crates/sherpa-rs). Enable with the `diarize` feature flag. Models are downloaded automatically on first use from the [sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx) project.

## Feature flags

| Flag | Description |
|------|-------------|
| `diarize` | Enable speaker diarization via sherpa-rs |
| `cuda` | GPU acceleration via CUDA (requires CUDA toolkit) |
| `metal` | GPU acceleration via Metal (macOS only) |
| `vulkan` | GPU acceleration via Vulkan |

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
  -m, --model <MODEL>        Path to the GGML/GGUF Whisper model file [env: WHISPER_MODEL]
  -l, --language <LANGUAGE>  Language code ("it", "en") or "auto" [default: auto]
  -t, --translate            Translate to English instead of transcribing
  -d, --diarize              Enable speaker diarization (requires --features diarize)
  -j, --threads <THREADS>    Number of inference threads [default: all cores]
```

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

## Whisper models

Download GGML models from [Hugging Face](https://huggingface.co/ggerganov/whisper.cpp/tree/main):

| Model | Size | Speed | Quality |
|-------|------|-------|---------|
| `ggml-tiny.bin` | 75 MB | Fastest | Basic |
| `ggml-base.bin` | 142 MB | Fast | Good |
| `ggml-small.bin` | 466 MB | Medium | Better |
| `ggml-medium.bin` | 1.5 GB | Slow | High |
| `ggml-large-v3-turbo.bin` | 1.6 GB | Medium | Best |

## Supported platforms

- Windows x86_64
- Windows ARM64 (Qualcomm Snapdragon X Elite)
- macOS (Apple Silicon / Intel)
- Linux x86_64

## License

AGPL-3.0-or-later — see [LICENSE](LICENSE) for details.

Copyright © 2026 Dario Finardi
