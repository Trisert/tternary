use std::fs::File;
use std::io::{BufWriter, Write};

use memmap2::Mmap;
use pyo3::exceptions::{PyIOError, PyValueError};
use pyo3::prelude::*;
use rayon::prelude::*;
use tokenizers::decoders::byte_level::ByteLevel as ByteLevelDecoder;
use tokenizers::models::bpe::BPE;
use tokenizers::models::bpe::trainer::BpeTrainer;
use tokenizers::pre_tokenizers::byte_level::ByteLevel;
use tokenizers::tokenizer::{DecoderWrapper, NormalizerWrapper, PostProcessorWrapper, PreTokenizerWrapper};
use tokenizers::Tokenizer as HFTokenizer;

#[pyclass(name = "Tokenizer")]
struct Tokenizer {
    inner: HFTokenizer,
}

#[pymethods]
impl Tokenizer {
    #[staticmethod]
    fn load(path: &str) -> PyResult<Self> {
        let mut tokenizer =
            HFTokenizer::from_file(path).map_err(|e| PyIOError::new_err(e.to_string()))?;
        tokenizer.with_decoder(Some(ByteLevelDecoder::new(true, false, false)));
        Ok(Tokenizer { inner: tokenizer })
    }

    fn encode(&self, text: &str) -> PyResult<Vec<u32>> {
        let encoding = self
            .inner
            .encode(text, true)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(encoding.get_ids().to_vec())
    }

    fn encode_batch(&self, texts: Vec<String>) -> PyResult<Vec<Vec<u32>>> {
        let refs: Vec<&str> = texts.iter().map(|s| s.as_str()).collect();
        let encodings = self
            .inner
            .encode_batch_fast(refs, false)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(encodings
            .into_iter()
            .map(|e| e.get_ids().to_vec())
            .collect())
    }

    fn decode(&self, ids: Vec<u32>, skip_special_tokens: bool) -> PyResult<String> {
        self.inner
            .decode(&ids, skip_special_tokens)
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    fn get_vocab_size(&self) -> usize {
        self.inner.get_vocab_size(true)
    }
}

#[pyfunction]
fn train_tokenizer(text_path: &str, output_path: &str, vocab_size: usize) -> PyResult<()> {
    let mut tokenizer: tokenizers::tokenizer::TokenizerImpl<
        BPE,
        NormalizerWrapper,
        PreTokenizerWrapper,
        PostProcessorWrapper,
        DecoderWrapper,
    > = tokenizers::tokenizer::TokenizerImpl::new(BPE::default());
    tokenizer.with_pre_tokenizer(Some(ByteLevel::new(false, false, true)));
    tokenizer.with_decoder(Some(ByteLevelDecoder::new(true, false, false)));

    let mut trainer = BpeTrainer::builder().vocab_size(vocab_size).build();
    tokenizer
        .train_from_files(&mut trainer, vec![text_path.to_string()])
        .map_err(|e| PyIOError::new_err(e.to_string()))?;
    tokenizer
        .save(output_path, true)
        .map_err(|e| PyIOError::new_err(e.to_string()))?;
    Ok(())
}

#[pyfunction]
#[pyo3(signature = (input_path, tokenizer_path, output_path, batch_size=None))]
fn tokenize_dataset_file(
    input_path: &str,
    tokenizer_path: &str,
    output_path: &str,
    batch_size: Option<usize>,
) -> PyResult<usize> {
    let batch_size = batch_size.unwrap_or(10000);

    let tokenizer =
        HFTokenizer::from_file(tokenizer_path).map_err(|e| PyIOError::new_err(e.to_string()))?;

    let file = File::open(input_path).map_err(|e| PyIOError::new_err(e.to_string()))?;
    let mmap = unsafe { Mmap::map(&file).map_err(|e| PyIOError::new_err(e.to_string()))? };
    let text = std::str::from_utf8(&mmap).map_err(|e| PyValueError::new_err(e.to_string()))?;

    let lines: Vec<&str> = text.lines().collect();

    let out_file =
        File::create(output_path).map_err(|e| PyIOError::new_err(e.to_string()))?;
    let mut writer = BufWriter::with_capacity(64 * 1024 * 1024, out_file);

    let chunks: Vec<Vec<&str>> = lines.chunks(batch_size).map(|c| c.to_vec()).collect();

    let encoded_chunks: Vec<Vec<Vec<u32>>> = chunks
        .par_iter()
        .map(|chunk| {
            tokenizer
                .encode_batch_fast(chunk.clone(), false)
                .unwrap()
                .into_iter()
                .map(|e| e.get_ids().to_vec())
                .collect()
        })
        .collect();

    let mut total_tokens = 0usize;
    for chunk in &encoded_chunks {
        for ids in chunk {
            for &id in ids {
                writer
                    .write_all(&(id as u16).to_le_bytes())
                    .map_err(|e| PyIOError::new_err(e.to_string()))?;
            }
            total_tokens += ids.len();
        }
    }
    writer
        .flush()
        .map_err(|e| PyIOError::new_err(e.to_string()))?;

    Ok(total_tokens)
}

#[pymodule]
fn pytternary_tokenizer(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Tokenizer>()?;
    m.add_function(wrap_pyfunction!(train_tokenizer, m)?)?;
    m.add_function(wrap_pyfunction!(tokenize_dataset_file, m)?)?;
    Ok(())
}
