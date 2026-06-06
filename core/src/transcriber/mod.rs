mod whisper_cpp;

pub use whisper_cpp::WhisperCppBackend;

/// A transcribed segment of speech with timing information.
#[derive(Debug, Clone)]
pub struct Segment {
    /// Start time in milliseconds.
    pub start_ms: i64,
    /// End time in milliseconds.
    pub end_ms: i64,
    /// Transcribed text.
    pub text: String,
}

/// Configuration for transcription.
#[derive(Debug, Clone)]
pub struct TranscribeParams {
    /// Language code (e.g. "it", "en", "auto" for detection).
    pub language: String,
    /// Whether to translate to English.
    pub translate: bool,
    /// Number of threads for inference.
    pub n_threads: usize,
}

impl Default for TranscribeParams {
    fn default() -> Self {
        Self {
            language: "auto".to_string(),
            translate: false,
            n_threads: num_cpus(),
        }
    }
}

/// Trait abstracting a speech-to-text backend.
pub trait TranscriberBackend {
    /// Transcribe a buffer of 16kHz mono f32 PCM samples.
    fn transcribe(&mut self, pcm_16khz: &[f32], params: &TranscribeParams) -> anyhow::Result<Vec<Segment>>;
}

fn num_cpus() -> usize {
    std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(4)
}
