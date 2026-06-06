use anyhow::{Context, Result};
use rubato::{Resampler, SincFixedIn, SincInterpolationParameters, SincInterpolationType, WindowFunction};

use super::WHISPER_SAMPLE_RATE;

/// Resample mono f32 audio to 16kHz. Returns the input unchanged if already at 16kHz.
pub fn resample_to_16khz(samples: &[f32], source_rate: u32) -> Result<Vec<f32>> {
    if source_rate == WHISPER_SAMPLE_RATE {
        return Ok(samples.to_vec());
    }

    let ratio = WHISPER_SAMPLE_RATE as f64 / source_rate as f64;
    let params = SincInterpolationParameters {
        sinc_len: 256,
        f_cutoff: 0.95,
        interpolation: SincInterpolationType::Linear,
        oversampling_factor: 256,
        window: WindowFunction::BlackmanHarris2,
    };

    let chunk_size = 1024;
    let mut resampler = SincFixedIn::<f32>::new(
        ratio,
        2.0,
        params,
        chunk_size,
        1,
    )
    .context("Failed to create resampler")?;

    let mut output = Vec::with_capacity((samples.len() as f64 * ratio) as usize + 1024);
    let mut pos = 0;

    while pos < samples.len() {
        let end = (pos + chunk_size).min(samples.len());
        let mut chunk = samples[pos..end].to_vec();

        if chunk.len() < chunk_size {
            chunk.resize(chunk_size, 0.0);
        }

        let result = resampler
            .process(&[chunk], None)
            .context("Resampling failed")?;
        output.extend_from_slice(&result[0]);
        pos += chunk_size;
    }

    let expected_len = (samples.len() as f64 * ratio) as usize;
    output.truncate(expected_len);

    log::info!("Resampled {}Hz -> {}Hz ({} samples)", source_rate, WHISPER_SAMPLE_RATE, output.len());
    Ok(output)
}
