use anyhow::{Context, Result, bail};
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use cpal::{SampleFormat, SampleRate, StreamConfig};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};

use crate::audio::resample_to_16khz;
use crate::transcriber::{Segment, TranscribeParams, TranscriberBackend};

/// Duration of each audio chunk sent to the transcriber (in seconds).
const CHUNK_DURATION_SECS: f32 = 5.0;

/// Overlap between consecutive chunks to avoid cutting words (in seconds).
const OVERLAP_SECS: f32 = 0.5;

/// Run live microphone transcription until Ctrl+C is pressed.
pub fn run_live<B: TranscriberBackend>(
    transcriber: &mut B,
    params: &TranscribeParams,
) -> Result<()> {
    let host = cpal::default_host();
    let device = host
        .default_input_device()
        .context("No input device available")?;

    let device_name = device.name().unwrap_or_else(|_| "unknown".into());
    log::info!("Using input device: {}", device_name);

    let supported_config = device
        .default_input_config()
        .context("No supported input config")?;

    let source_rate = supported_config.sample_rate().0;
    let channels = supported_config.channels() as usize;
    let sample_format = supported_config.sample_format();

    log::info!(
        "Capturing at {}Hz, {} channels, {:?}",
        source_rate,
        channels,
        sample_format,
    );

    let config = StreamConfig {
        channels: channels as u16,
        sample_rate: SampleRate(source_rate),
        buffer_size: cpal::BufferSize::Default,
    };

    let buffer: Arc<Mutex<Vec<f32>>> = Arc::new(Mutex::new(Vec::new()));
    let buffer_clone = Arc::clone(&buffer);

    let running = Arc::new(AtomicBool::new(true));
    ctrlc_flag(Arc::clone(&running));

    let err_fn = |err| log::error!("Audio stream error: {}", err);

    let stream = match sample_format {
        SampleFormat::F32 => device.build_input_stream(
            &config,
            move |data: &[f32], _| {
                let mono = mix_to_mono(data, channels);
                buffer_clone.lock().unwrap().extend_from_slice(&mono);
            },
            err_fn,
            None,
        ),
        SampleFormat::I16 => device.build_input_stream(
            &config,
            move |data: &[i16], _| {
                let floats: Vec<f32> = data.iter().map(|&s| s as f32 / i16::MAX as f32).collect();
                let mono = mix_to_mono(&floats, channels);
                buffer_clone.lock().unwrap().extend_from_slice(&mono);
            },
            err_fn,
            None,
        ),
        _ => bail!("Unsupported sample format: {:?}", sample_format),
    }
    .context("Failed to build input stream")?;

    stream.play().context("Failed to start audio stream")?;

    let chunk_samples = (source_rate as f32 * CHUNK_DURATION_SECS) as usize;
    let overlap_samples = (source_rate as f32 * OVERLAP_SECS) as usize;

    println!("Listening... Press Ctrl+C to stop.\n");

    while running.load(Ordering::Relaxed) {
        std::thread::sleep(std::time::Duration::from_millis(200));

        let available = buffer.lock().unwrap().len();
        if available < chunk_samples {
            continue;
        }

        let chunk: Vec<f32> = {
            let mut buf = buffer.lock().unwrap();
            let drain_end = chunk_samples;
            let chunk = buf[..drain_end].to_vec();
            let keep_from = drain_end.saturating_sub(overlap_samples);
            *buf = buf[keep_from..].to_vec();
            chunk
        };

        let pcm_16k = resample_to_16khz(&chunk, source_rate)?;

        match transcriber.transcribe(&pcm_16k, params) {
            Ok(segments) => print_segments(&segments),
            Err(e) => log::warn!("Transcription error: {}", e),
        }
    }

    let remaining: Vec<f32> = {
        let buf = buffer.lock().unwrap();
        buf.clone()
    };
    if remaining.len() > (source_rate as usize / 2) {
        let pcm_16k = resample_to_16khz(&remaining, source_rate)?;
        if let Ok(segments) = transcriber.transcribe(&pcm_16k, params) {
            print_segments(&segments);
        }
    }

    println!("\nStopped.");
    Ok(())
}

fn mix_to_mono(interleaved: &[f32], channels: usize) -> Vec<f32> {
    if channels == 1 {
        return interleaved.to_vec();
    }
    interleaved
        .chunks_exact(channels)
        .map(|ch| ch.iter().sum::<f32>() / channels as f32)
        .collect()
}

fn print_segments(segments: &[Segment]) {
    for seg in segments {
        let text = seg.text.trim();
        if !text.is_empty() {
            println!("{}", text);
        }
    }
}

fn ctrlc_flag(running: Arc<AtomicBool>) {
    let _ = ctrlc::set_handler(move || {
        running.store(false, Ordering::Relaxed);
    });
}
