use std::path::PathBuf;

use anyhow::{Context, Result};
use clap::{Parser, Subcommand};

use scriba_core::audio::{decode_audio_file, resample_to_16khz};
use scriba_core::mic;
use scriba_core::transcriber::{TranscribeParams, TranscriberBackend, WhisperCppBackend};

#[derive(Parser)]
#[command(name = "whisper-cli", version, about = "Speech-to-text powered by Whisper")]
struct Cli {
    /// Path to the GGML/GGUF Whisper model file.
    #[arg(short, long, env = "WHISPER_MODEL")]
    model: PathBuf,

    /// Language code (e.g. "it", "en") or "auto" for detection.
    #[arg(short, long, default_value = "auto")]
    language: String,

    /// Translate to English instead of transcribing.
    #[arg(short, long, default_value_t = false)]
    translate: bool,

    /// Enable speaker diarization (requires diarize feature and models).
    #[arg(short, long, default_value_t = false)]
    diarize: bool,

    /// Number of threads (default: all available).
    #[arg(short = 'j', long)]
    threads: Option<usize>,

    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Transcribe an audio file (wav, mp3, ogg).
    File {
        /// Path to the audio file.
        path: PathBuf,
    },
    /// Live transcription from the system microphone.
    Listen,
}

fn main() -> Result<()> {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info"))
        .format_timestamp_millis()
        .init();

    let cli = Cli::parse();

    let params = TranscribeParams {
        language: cli.language.clone(),
        translate: cli.translate,
        n_threads: cli.threads.unwrap_or_else(|| {
            std::thread::available_parallelism()
                .map(|n| n.get())
                .unwrap_or(4)
        }),
    };

    let mut backend = WhisperCppBackend::new(&cli.model)
        .context("Failed to initialize Whisper backend")?;

    match cli.command {
        Command::File { path } => transcribe_file(&mut backend, &path, &params, cli.diarize),
        Command::Listen => mic::run_live(&mut backend, &params),
    }
}

fn transcribe_file(
    backend: &mut WhisperCppBackend,
    path: &PathBuf,
    params: &TranscribeParams,
    diarize: bool,
) -> Result<()> {
    let decoded = decode_audio_file(path)?;
    let pcm_16k = resample_to_16khz(&decoded.samples, decoded.sample_rate)?;

    log::info!(
        "Transcribing {:.1}s of audio...",
        pcm_16k.len() as f64 / 16000.0
    );

    let segments = backend.transcribe(&pcm_16k, params)?;

    if segments.is_empty() {
        println!("(no speech detected)");
        return Ok(());
    }

    #[cfg(feature = "diarize")]
    if diarize {
        return transcribe_with_diarize(&segments, &pcm_16k);
    }

    #[cfg(not(feature = "diarize"))]
    if diarize {
        anyhow::bail!("Diarize feature not enabled at compile time. Rebuild with --features diarize");
    }

    for seg in &segments {
        let start = format_timestamp(seg.start_ms);
        let end = format_timestamp(seg.end_ms);
        println!("[{} --> {}]  {}", start, end, seg.text.trim());
    }

    Ok(())
}

#[cfg(feature = "diarize")]
fn transcribe_with_diarize(
    segments: &[scriba_core::transcriber::Segment],
    pcm_16k: &[f32],
) -> Result<()> {
    use scriba_core::diarize;

    if !diarize::models_installed() {
        log::info!("Diarization models not found. Downloading...");
        diarize::download_models(|dl, total| {
            if let Some(t) = total {
                eprint!("\r  download: {} / {} ({:.0}%)", dl, t, dl as f64 / t as f64 * 100.0);
            }
        })?;
        eprintln!();
    }

    log::info!("Running speaker diarization...");
    let speaker_segs = diarize::diarize(pcm_16k, None)?;
    log::info!("Found {} speaker segments", speaker_segs.len());

    let transcript_tuples: Vec<(f32, f32, String)> = segments
        .iter()
        .map(|s| (s.start_ms as f32 / 1000.0, s.end_ms as f32 / 1000.0, s.text.clone()))
        .collect();

    let merged = diarize::merge_transcript_with_speakers(&transcript_tuples, &speaker_segs);
    println!("{merged}");

    println!("\n--- Detailed segments ---");
    for seg in segments {
        let start = format_timestamp(seg.start_ms);
        let end = format_timestamp(seg.end_ms);
        let mid = (seg.start_ms as f32 + seg.end_ms as f32) / 2000.0;
        let speaker = speaker_segs
            .iter()
            .find(|s| s.start <= mid && mid <= s.end)
            .map(|s| format!("Speaker {}", s.speaker + 1))
            .unwrap_or_else(|| "?".to_string());
        println!("[{} --> {}] [{}]  {}", start, end, speaker, seg.text.trim());
    }

    Ok(())
}

fn format_timestamp(ms: i64) -> String {
    let total_secs = ms / 1000;
    let minutes = total_secs / 60;
    let seconds = total_secs % 60;
    let millis = ms % 1000;
    format!("{:02}:{:02}.{:03}", minutes, seconds, millis)
}
