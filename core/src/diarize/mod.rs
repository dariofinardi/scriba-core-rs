use std::path::{Path, PathBuf};

/// A speaker-labeled time segment.
#[derive(Debug, Clone)]
pub struct SpeakerSegment {
    pub start: f32,
    pub end: f32,
    pub speaker: i32,
}

pub struct DiarizeModels {
    pub segmentation: PathBuf,
    pub embedding: PathBuf,
}

const SEG_MODEL: &str = "sherpa-onnx-pyannote-segmentation-3-0/model.onnx";
const EMB_MODEL: &str = "3dspeaker_speech_eres2net_base_sv_zh-cn_3dspeaker_16k.onnx";

const SEG_URL: &str = "https://github.com/k2-fsa/sherpa-onnx/releases/download/speaker-segmentation-models/sherpa-onnx-pyannote-segmentation-3-0.tar.bz2";
const EMB_URL: &str = "https://github.com/k2-fsa/sherpa-onnx/releases/download/speaker-recongition-models/3dspeaker_speech_eres2net_base_sv_zh-cn_3dspeaker_16k.onnx";

fn default_models_dir() -> PathBuf {
    let mut dir = dirs::data_dir().unwrap_or_else(|| PathBuf::from("."));
    dir.push("scriba");
    dir.push("diarize");
    dir
}

pub fn models_dir_with_base(data_dir: Option<&Path>) -> PathBuf {
    match data_dir {
        Some(d) => d.join("diarize"),
        None => default_models_dir(),
    }
}

pub fn models_dir() -> PathBuf {
    default_models_dir()
}

pub fn segmentation_model_path() -> PathBuf {
    models_dir().join(SEG_MODEL)
}

pub fn embedding_model_path() -> PathBuf {
    models_dir().join(EMB_MODEL)
}

pub fn models_installed() -> bool {
    segmentation_model_path().exists() && embedding_model_path().exists()
}

pub fn models_installed_at(data_dir: Option<&Path>) -> bool {
    let dir = models_dir_with_base(data_dir);
    dir.join(SEG_MODEL).exists() && dir.join(EMB_MODEL).exists()
}

/// Download diarization models. Reports progress as (downloaded, total).
pub fn download_models<F: FnMut(u64, Option<u64>)>(mut progress: F) -> anyhow::Result<()> {
    download_models_to(&models_dir(), &mut progress)
}

pub fn download_models_to<F: FnMut(u64, Option<u64>)>(dir: &Path, progress: &mut F) -> anyhow::Result<()> {
    use std::io::{Read, Write};

    std::fs::create_dir_all(dir)?;

    let emb_path = dir.join(EMB_MODEL);
    if !emb_path.exists() {
        eprintln!("[scriba] Downloading speaker embedding model...");
        let mut resp = reqwest::blocking::get(EMB_URL)?.error_for_status()?;
        let total = resp.content_length();
        let tmp = dir.join("embedding.part");
        let mut file = std::fs::File::create(&tmp)?;
        let mut buf = [0u8; 65536];
        let mut downloaded = 0u64;
        loop {
            let n = resp.read(&mut buf)?;
            if n == 0 { break; }
            file.write_all(&buf[..n])?;
            downloaded += n as u64;
            progress(downloaded, total);
        }
        file.flush()?;
        std::fs::rename(&tmp, &emb_path)?;
    }

    let seg_path = dir.join(SEG_MODEL);
    if !seg_path.exists() {
        eprintln!("[scriba] Downloading speaker segmentation model...");
        let archive_path = dir.join("seg.tar.bz2");
        let mut resp = reqwest::blocking::get(SEG_URL)?.error_for_status()?;
        let total = resp.content_length();
        let mut file = std::fs::File::create(&archive_path)?;
        let mut buf = [0u8; 65536];
        let mut downloaded = 0u64;
        loop {
            let n = resp.read(&mut buf)?;
            if n == 0 { break; }
            file.write_all(&buf[..n])?;
            downloaded += n as u64;
            progress(downloaded, total.map(|t| t + 15_000_000));
        }
        file.flush()?;
        drop(file);

        extract_tar_bz2(&archive_path, dir)?;
        let _ = std::fs::remove_file(&archive_path);
    }

    Ok(())
}

fn extract_tar_bz2(archive: &Path, dest: &Path) -> anyhow::Result<()> {
    use std::io::BufReader;

    let file = std::fs::File::open(archive)?;
    let reader = BufReader::new(file);
    let decompressor = bzip2::read::BzDecoder::new(reader);
    let mut archive = tar::Archive::new(decompressor);
    archive.unpack(dest)?;
    Ok(())
}

/// Run speaker diarization on 16kHz mono f32 audio.
pub fn diarize(
    samples_16khz: &[f32],
    progress_cb: Option<Box<dyn Fn(i32, i32) -> i32 + Send + 'static>>,
) -> anyhow::Result<Vec<SpeakerSegment>> {
    let seg_path = segmentation_model_path();
    let emb_path = embedding_model_path();

    if !seg_path.exists() || !emb_path.exists() {
        anyhow::bail!("Diarization models not installed. Download them from Setup.");
    }

    let seg_str = seg_path.to_str().ok_or_else(|| anyhow::anyhow!("invalid path"))?;
    let emb_str = emb_path.to_str().ok_or_else(|| anyhow::anyhow!("invalid path"))?;

    let config = sherpa_rs::diarize::DiarizeConfig {
        num_clusters: None,
        ..Default::default()
    };

    let mut sd = sherpa_rs::diarize::Diarize::new(seg_str, emb_str, config)
        .map_err(|e| anyhow::anyhow!("{e}"))?;

    let segments = sd.compute(samples_16khz.to_vec(), progress_cb)
        .map_err(|e| anyhow::anyhow!("{e}"))?;

    Ok(segments
        .into_iter()
        .map(|s| SpeakerSegment {
            start: s.start,
            end: s.end,
            speaker: s.speaker,
        })
        .collect())
}

/// Merge whisper transcript segments with speaker diarization segments.
pub fn merge_transcript_with_speakers(
    transcript_segments: &[(f32, f32, String)],
    speaker_segments: &[SpeakerSegment],
) -> String {
    let mut result = String::new();
    let mut last_speaker = -1;

    for (t_start, t_end, text) in transcript_segments {
        let text = text.trim();
        if text.is_empty() { continue; }

        let mid = (t_start + t_end) / 2.0;
        let speaker = speaker_segments
            .iter()
            .find(|s| s.start <= mid && mid <= s.end)
            .map(|s| s.speaker)
            .unwrap_or(-1);

        if speaker != last_speaker && speaker >= 0 {
            if !result.is_empty() { result.push('\n'); }
            result.push_str(&format!("Speaker {}: ", speaker + 1));
            last_speaker = speaker;
        }
        result.push_str(text);
        result.push(' ');
    }

    result.trim().to_string()
}
