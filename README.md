# scriba-core

**Motore di trascrizione vocale headless per Rust** — decodifica audio, resampling, inferenza Whisper, cattura microfono e diarizzazione speaker, senza alcuna dipendenza da UI.

![License](https://img.shields.io/badge/license-AGPL--3.0--or--later-blue)
![Platform](https://img.shields.io/badge/platform-Windows%20%7C%20macOS%20%7C%20Linux-lightgrey)

## Perché scriba-core

I servizi di trascrizione cloud presentano limiti concreti: latenza di rete, costi a consumo, rischi per la privacy dei dati vocali. `scriba-core` porta la trascrizione interamente on-device, utilizzando [whisper.cpp](https://github.com/ggerganov/whisper.cpp) — l'implementazione C/C++ di [Whisper](https://github.com/openai/whisper) di OpenAI — attraverso binding Rust nativi.

Il crate è pensato come fondazione riutilizzabile: è il cuore di [Scriba](https://github.com/dariofinardi/scriba-rs) (l'app desktop con UI), ma può essere integrato in qualsiasi applicazione Rust che necessiti di speech-to-text senza dipendere da un framework grafico.

## Struttura del workspace

```
scriba-core-rs/
├── core/          # Libreria scriba-core
├── whisper-cli/   # Tool CLI per trascrizione da terminale
└── cmake/         # Helper di build per cross-compilazione ARM64
```

## Funzionalità

### Decodifica audio
Decodifica file WAV, MP3 e OGG in PCM mono f32 tramite [Symphonia](https://crates.io/crates/symphonia). L'audio multi-canale viene automaticamente mixato in mono.

### Resampling
Resampling sinc di alta qualità a 16 kHz (il sample rate atteso da Whisper) via [Rubato](https://crates.io/crates/rubato), con filtro finestra BlackmanHarris2 a 256 tap. Passthrough quando la sorgente è già a 16 kHz.

### Trascrizione
Inferenza whisper.cpp attraverso il trait `TranscriberBackend`, che restituisce `Segment` con timestamp di inizio e fine. Supporta:
- Rilevamento automatico della lingua o selezione esplicita
- Traduzione verso l'inglese
- Numero di thread configurabile

### Cattura microfono in tempo reale
Trascrizione live dal microfono di sistema via [cpal](https://crates.io/crates/cpal). L'audio viene catturato in finestre con overlap per evitare il taglio di parole ai confini dei chunk.

### Diarizzazione speaker (opzionale)
Identifica *chi* sta parlando in ogni segmento tramite [sherpa-rs](https://crates.io/crates/sherpa-rs). Si attiva con la feature flag `diarize`. I modelli di segmentazione ed embedding vengono scaricati automaticamente al primo utilizzo dal progetto [sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx).

## Feature flag

| Flag | Descrizione |
|------|-------------|
| `diarize` | Diarizzazione speaker via sherpa-rs + ONNX Runtime |
| `cuda` | Accelerazione GPU NVIDIA (CUDA) |
| `metal` | Accelerazione GPU Apple (Metal) |
| `vulkan` | Accelerazione GPU cross-platform (Vulkan) |

## Utilizzo come libreria

Aggiungi al tuo `Cargo.toml`:

```toml
[dependencies]
scriba-core = { git = "https://github.com/dariofinardi/scriba-core-rs.git", tag = "v20260606" }
```

### Trascrivere un file

```rust
use scriba_core::audio::{decode_audio_file, resample_to_16khz};
use scriba_core::transcriber::{TranscribeParams, TranscriberBackend, WhisperCppBackend};

let decoded = decode_audio_file("registrazione.wav".as_ref())?;
let pcm = resample_to_16khz(&decoded.samples, decoded.sample_rate)?;

let mut backend = WhisperCppBackend::new("ggml-large-v3-turbo.bin".as_ref())?;
let segments = backend.transcribe(&pcm, &TranscribeParams::default())?;

for seg in &segments {
    println!("[{} → {}] {}", seg.start_ms, seg.end_ms, seg.text.trim());
}
```

### Trascrizione live dal microfono

```rust
use scriba_core::mic;
use scriba_core::transcriber::{TranscribeParams, WhisperCppBackend};

let mut backend = WhisperCppBackend::new("modello.bin".as_ref())?;
mic::run_live(&mut backend, &TranscribeParams::default())?;
```

## whisper-cli

Tool da riga di comando incluso nel workspace, pronto all'uso.

### Trascrizione file

```sh
cargo run -p whisper-cli --release -- --model ggml-large-v3-turbo.bin file registrazione.wav
```

### Microfono live

```sh
cargo run -p whisper-cli --release -- --model ggml-large-v3-turbo.bin listen
```

### Con diarizzazione speaker

```sh
cargo run -p whisper-cli --release --features diarize -- --model ggml-large-v3-turbo.bin --diarize file riunione.wav
```

### Opzioni

```
Usage: whisper-cli [OPTIONS] --model <MODEL> <COMMAND>

Commands:
  file    Trascrivi un file audio (wav, mp3, ogg)
  listen  Trascrizione live dal microfono di sistema

Options:
  -m, --model <MODEL>        Path al file modello GGML/GGUF [env: WHISPER_MODEL]
  -l, --language <LANGUAGE>  Codice lingua ("it", "en") o "auto" [default: auto]
  -t, --translate            Traduci in inglese invece di trascrivere
  -d, --diarize              Abilita diarizzazione (richiede --features diarize)
  -j, --threads <THREADS>    Numero thread di inferenza [default: tutti i core]
```

## Modelli Whisper

I modelli GGML si scaricano da [Hugging Face](https://huggingface.co/ggerganov/whisper.cpp/tree/main). Scriba usa versioni quantizzate Q5 per ridurre la dimensione mantenendo qualità prossima al float16:

| Modello | Dimensione | Velocità | Qualità | Uso consigliato |
|---------|-----------|----------|---------|-----------------|
| `ggml-small-q5_1.bin` | 190 MB | ★★★★ | ★★ | Appunti rapidi, bozze |
| `ggml-medium-q5_0.bin` | 515 MB | ★★★ | ★★★ | Buon compromesso |
| `ggml-large-v3-turbo-q5_0.bin` | 574 MB | ★★★ | ★★★★ | Massima accuratezza |

Tutti supportano oltre 90 lingue.

## Build

### Build standard

```sh
cargo build --release
```

### Con diarizzazione

```sh
cargo build --release --features diarize
```

### Windows ARM64 (Qualcomm Snapdragon)

Richiede Ninja e clang-cl. Impostare le variabili d'ambiente:

```powershell
$env:PATH = "cmake;" + $env:PATH
$env:CMAKE_TOOLCHAIN_FILE = "cmake/arm64-toolchain.cmake"
$env:CMAKE_GENERATOR = "Ninja"
$env:CMAKE_C_COMPILER = "clang-cl"
$env:CMAKE_CXX_COMPILER = "clang-cl"
$env:CMAKE_ASM_COMPILER = "clang-cl"
```

Poi compilare normalmente con `cargo build --release`.

## Dipendenze principali

| Crate | Ruolo |
|-------|-------|
| [whisper-rs](https://github.com/dariofinardi/whisper-rs) | Binding Rust per whisper.cpp (fork con fix Windows) |
| [symphonia](https://crates.io/crates/symphonia) | Decodifica audio multi-formato |
| [rubato](https://crates.io/crates/rubato) | Resampling sinc di alta qualità |
| [cpal](https://crates.io/crates/cpal) | Cattura audio cross-platform |
| [sherpa-rs](https://crates.io/crates/sherpa-rs) | Diarizzazione speaker (opzionale) |

## Piattaforme supportate

- Windows x86_64
- Windows ARM64 (Qualcomm Snapdragon X Elite)
- macOS (Apple Silicon / Intel)
- Linux x86_64

## Licenza

AGPL-3.0-or-later — vedi [LICENSE](LICENSE).

Copyright © 2026 Dario Finardi
