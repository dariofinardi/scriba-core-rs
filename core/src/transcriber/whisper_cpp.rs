use anyhow::{Context, Result};
use std::path::Path;
use whisper_rs::{FullParams, SamplingStrategy, WhisperContext, WhisperContextParameters};

use super::{Segment, TranscribeParams, TranscriberBackend};

/// Backend using whisper.cpp via whisper-rs bindings.
pub struct WhisperCppBackend {
    ctx: WhisperContext,
}

impl WhisperCppBackend {
    /// Load a GGML/GGUF model from disk.
    pub fn new(model_path: &Path) -> Result<Self> {
        let params = WhisperContextParameters::default();
        let ctx = WhisperContext::new_with_params(model_path, params)
            .context("Failed to load Whisper model")?;
        log::info!("Loaded model from {}", model_path.display());
        Ok(Self { ctx })
    }
}

impl TranscriberBackend for WhisperCppBackend {
    fn transcribe(&mut self, pcm_16khz: &[f32], params: &TranscribeParams) -> Result<Vec<Segment>> {
        if pcm_16khz.is_empty() {
            return Ok(Vec::new());
        }

        let mut state = self.ctx.create_state().context("Failed to create whisper state")?;

        let mut full_params = FullParams::new(SamplingStrategy::Greedy { best_of: 1 });
        full_params.set_n_threads(params.n_threads as i32);
        full_params.set_translate(params.translate);
        full_params.set_print_progress(false);
        full_params.set_print_realtime(false);
        full_params.set_print_timestamps(false);
        full_params.set_print_special(false);
        full_params.set_single_segment(false);

        if params.language == "auto" {
            full_params.set_detect_language(true);
        } else {
            full_params.set_language(Some(&params.language));
        }

        state
            .full(full_params, pcm_16khz)
            .context("Whisper inference failed")?;

        let mut segments = Vec::new();
        for seg in state.as_iter() {
            let text = seg.to_str_lossy().unwrap_or_default().to_string();
            segments.push(Segment {
                start_ms: seg.start_timestamp() * 10,
                end_ms: seg.end_timestamp() * 10,
                text,
            });
        }

        Ok(segments)
    }
}
