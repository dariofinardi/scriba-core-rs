mod decode;
mod resample;

pub use decode::{decode_audio_file, DecodedAudio};
pub use resample::resample_to_16khz;

/// Target sample rate for Whisper models.
pub const WHISPER_SAMPLE_RATE: u32 = 16_000;
