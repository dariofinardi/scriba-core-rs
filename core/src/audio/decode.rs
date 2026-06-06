use anyhow::{Context, Result, bail};
use std::path::Path;
use symphonia::core::audio::SampleBuffer;
use symphonia::core::codecs::DecoderOptions;
use symphonia::core::formats::FormatOptions;
use symphonia::core::io::MediaSourceStream;
use symphonia::core::meta::MetadataOptions;
use symphonia::core::probe::Hint;

/// Decoded audio: mono f32 samples at the original sample rate.
pub struct DecodedAudio {
    pub samples: Vec<f32>,
    pub sample_rate: u32,
}

/// Decode an audio file (wav, mp3, ogg) into mono f32 PCM.
pub fn decode_audio_file(path: &Path) -> Result<DecodedAudio> {
    let file = std::fs::File::open(path)
        .with_context(|| format!("Cannot open audio file: {}", path.display()))?;
    let mss = MediaSourceStream::new(Box::new(file), Default::default());

    let mut hint = Hint::new();
    if let Some(ext) = path.extension().and_then(|e| e.to_str()) {
        hint.with_extension(ext);
    }

    let probed = symphonia::default::get_probe()
        .format(&hint, mss, &FormatOptions::default(), &MetadataOptions::default())
        .context("Unsupported or invalid audio format")?;

    let mut format = probed.format;

    let track = format
        .default_track()
        .context("No audio track found")?;
    let track_id = track.id;
    let sample_rate = track
        .codec_params
        .sample_rate
        .context("Unknown sample rate")?;
    let channels = track
        .codec_params
        .channels
        .map(|c| c.count())
        .unwrap_or(1);

    let mut decoder = symphonia::default::get_codecs()
        .make(&track.codec_params, &DecoderOptions::default())
        .context("Failed to create audio decoder")?;

    let mut all_samples: Vec<f32> = Vec::new();

    loop {
        let packet = match format.next_packet() {
            Ok(p) => p,
            Err(symphonia::core::errors::Error::IoError(ref e))
                if e.kind() == std::io::ErrorKind::UnexpectedEof =>
            {
                break;
            }
            Err(e) => bail!("Error reading packet: {e}"),
        };

        if packet.track_id() != track_id {
            continue;
        }

        let decoded = decoder.decode(&packet).context("Decode error")?;
        let spec = *decoded.spec();
        let duration = decoded.capacity();

        let mut sample_buf = SampleBuffer::<f32>::new(duration as u64, spec);
        sample_buf.copy_interleaved_ref(decoded);
        let interleaved = sample_buf.samples();

        if channels == 1 {
            all_samples.extend_from_slice(interleaved);
        } else {
            for chunk in interleaved.chunks_exact(channels) {
                let mono: f32 = chunk.iter().sum::<f32>() / channels as f32;
                all_samples.push(mono);
            }
        }
    }

    log::info!(
        "Decoded {} — {:.1}s, {}Hz, {} ch -> mono",
        path.display(),
        all_samples.len() as f64 / sample_rate as f64,
        sample_rate,
        channels,
    );

    Ok(DecodedAudio {
        samples: all_samples,
        sample_rate,
    })
}
