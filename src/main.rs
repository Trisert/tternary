use tternary::{Config, TernaryTransformer, Dataset};
use ndarray::Array2;
use rand::Rng;
use std::env;
use std::fs::File;
use std::io::{BufReader, Read, Write};
use std::time::Instant;
use tokenizers::models::bpe::BPE;
use tokenizers::pre_tokenizers::byte_level::ByteLevel;
use tokenizers::models::bpe::trainer::BpeTrainer;
use tokenizers::tokenizer::{TokenizerImpl, NormalizerWrapper, PreTokenizerWrapper, PostProcessorWrapper, DecoderWrapper};
use tokenizers::Tokenizer;
use ahash::AHashMap;

const ENCODED_FILE: &str = "data/tinystories_encoded.bin";
const TOKENIZER_FILE: &str = "data/tokenizer.json";
const BPE_VOCAB_SIZE: usize = 4096;
const ENCODE_CHUNK: usize = 64 * 1024 * 1024;

type BoxedError = Box<dyn std::error::Error + Send + Sync>;

fn train_tokenizer(text_path: &str) -> Result<Tokenizer, BoxedError> {
    println!("Training BPE tokenizer (vocab_size={})...", BPE_VOCAB_SIZE);
    let mut tokenizer: TokenizerImpl<BPE, NormalizerWrapper, PreTokenizerWrapper, PostProcessorWrapper, DecoderWrapper> = TokenizerImpl::new(BPE::new(AHashMap::new(), Vec::new()));
    tokenizer.with_pre_tokenizer(Some(ByteLevel::new(false, false, true)));

    let mut trainer = BpeTrainer::builder()
        .vocab_size(BPE_VOCAB_SIZE)
        .build();
    tokenizer.train_from_files(&mut trainer, vec![text_path.to_string()])?;
    println!("  Training complete, saving...");
    tokenizer.save(TOKENIZER_FILE, true)?;
    println!("  Saved to {}", TOKENIZER_FILE);
    println!("Tokenizer saved to {}", TOKENIZER_FILE);
    let trained = Tokenizer::from_file(TOKENIZER_FILE)?;
    Ok(trained)
}

fn encode_dataset(tokenizer: &Tokenizer, text_path: &str) -> Result<usize, BoxedError> {
    std::fs::create_dir_all("data/")?;
    let file = File::open(text_path)?;
    let mut reader = BufReader::new(file);
    let mut out = File::create(ENCODED_FILE)?;
    let mut total_tokens = 0usize;
    let mut leftover: Vec<u8> = Vec::new();
    let mut chunk_buf = vec![0u8; ENCODE_CHUNK];
    const LINES_PER_BATCH: usize = 10_000;

    loop {
        let n = reader.read(&mut chunk_buf)?;
        if n == 0 && leftover.is_empty() { break; }

        leftover.extend_from_slice(&chunk_buf[..n]);
        let valid_end = find_utf8_boundary(&leftover);
        let text = std::str::from_utf8(&leftover[..valid_end])?;

        for chunk in text.lines().collect::<Vec<_>>().chunks(LINES_PER_BATCH) {
            let encodings = tokenizer.encode_batch_fast(chunk.to_vec(), false)?;
            for encoding in encodings {
                for &id in encoding.get_ids() {
                    out.write_all(&id.to_le_bytes())?;
                }
                total_tokens += encoding.get_ids().len();
            }
        }

        if total_tokens % 50_000_000 == 0 {
            print!("\r  Encoded {}M tokens...", total_tokens / 1_000_000);
            std::io::stdout().flush()?;
        }

        leftover = leftover[valid_end..].to_vec();
        if n == 0 { break; }
    }
    println!("\r  Encoded {}M tokens.      ", total_tokens / 1_000_000);
    Ok(total_tokens)
}

fn find_utf8_boundary(bytes: &[u8]) -> usize {
    let mut i = bytes.len();
    while i > 0 {
        i -= 1;
        if (bytes[i] & 0xC0) != 0x80 {
            let byte = bytes[i];
            if byte & 0x80 == 0 {
                return i + 1;
            }
            let len = if byte & 0xE0 == 0xC0 { 2 }
                else if byte & 0xF0 == 0xE0 { 3 }
                else if byte & 0xF8 == 0xF0 { 4 }
                else { 1 };
            if i + len <= bytes.len() {
                return i + len;
            }
            return i;
        }
    }
    0
}

fn prepare_dataset() -> Result<(String, usize, Tokenizer), BoxedError> {
    std::fs::create_dir_all("data/")?;
    if std::path::Path::new(ENCODED_FILE).exists() && std::path::Path::new(TOKENIZER_FILE).exists() {
        let len = std::fs::metadata(ENCODED_FILE)?.len() as usize / 4;
        println!("Found cached encoded dataset ({} tokens)", len);
        let tokenizer = Tokenizer::from_file(TOKENIZER_FILE)?;
        return Ok((ENCODED_FILE.to_string(), len, tokenizer));
    }

    println!("Downloading TinyStories dataset from HuggingFace...");
    let api = hf_hub::api::sync::ApiBuilder::new()
        .with_progress(true)
        .build()?;
    let repo = api.dataset("roneneldan/TinyStories".to_string());
    let text_path = repo.download("TinyStories-train.txt")?;
    println!("  Downloaded to: {:?}", text_path);
    let text_path_str = text_path.to_string_lossy().to_string();

    let tokenizer = if std::path::Path::new(TOKENIZER_FILE).exists() {
        println!("Loading cached tokenizer...");
        Tokenizer::from_file(TOKENIZER_FILE)?
    } else {
        train_tokenizer(&text_path_str)?
    };

    println!("Tokenizer trained. Saving...");
    let vocab_size = tokenizer.get_vocab_size(true);
    println!("Tokenizer vocab size: {}", vocab_size);

    println!("Encoding dataset...");
    let num_tokens = encode_dataset(&tokenizer, &text_path_str)?;

    Ok((ENCODED_FILE.to_string(), num_tokens, tokenizer))
}

fn generate_sample(model: &mut TernaryTransformer, tokenizer: &Tokenizer, num_tokens: usize) {
    let vocab_size = tokenizer.get_vocab_size(true);
    let mut current_idx = 2u32;
    let temperature = 0.8;
    let mut generated = Vec::with_capacity(num_tokens);

    for _ in 0..num_tokens {
        let input_arr = Array2::from_elem((1, 1), current_idx as f32);
        let logits = model.forward(&input_arr);
        let cols = logits.shape()[1].min(vocab_size);

        let mut max_val = f32::NEG_INFINITY;
        for v in 0..cols {
            let val = logits[[0, v]] / temperature;
            if val > max_val { max_val = val; }
        }
        let sum_exp: f32 = (0..cols).map(|v| ((logits[[0, v]] / temperature) - max_val).exp()).sum();

        let r: f32 = rand::thread_rng().gen_range(0.0f32..1.0f32) * sum_exp;
        let mut cumsum = 0.0_f32;
        current_idx = (cols - 1) as u32;
        for (i, v) in (0..cols).enumerate() {
            let prob = ((logits[[0, v]] / temperature) - max_val).exp();
            cumsum += prob;
            if r <= cumsum { current_idx = i as u32; break; }
        }

        generated.push(current_idx);
    }

    let text = tokenizer.decode(&generated, true).unwrap_or_else(|_| "<decode error>".to_string());
    let truncated: String = text.chars().take(500).collect();
    println!("{}", truncated);
}

fn train() {
    let config = Config::default();
    println!("Config: embed_dim={}, hidden={}, layers={}, max_seq={}, kernel={}",
             config.embed_dim, config.hidden_dim,
             config.num_layers, config.max_seq_len, config.kernel_size);

    let (encoded_path, num_tokens, tokenizer) = prepare_dataset().expect("Failed to prepare dataset");
    let vocab_size = tokenizer.get_vocab_size(true);
    println!("Vocabulary size: {}", vocab_size);

    let config = Config { vocab_size, ..Config::default() };
    let batch_size = config.batch_size;
    let max_seq_len = config.max_seq_len;

    let mut dataset = Dataset::open(&encoded_path, max_seq_len);
    println!("Dataset: {} tokens ({:.1} MB on disk)", num_tokens, num_tokens as f64 * 4.0 / 1e6);

    let mut model = TernaryTransformer::new(config);
    println!("Parameters: {}", model.num_parameters());

    let args: Vec<String> = env::args().collect();
    let num_epochs = args.iter()
        .position(|a| a == "--epochs")
        .and_then(|i| args.get(i + 1)?.parse().ok())
        .unwrap_or(10);
    let steps_per_epoch: usize = args.iter()
        .position(|a| a == "--steps")
        .and_then(|i| args.get(i + 1)?.parse().ok())
        .unwrap_or(500);
    let lr = args.iter()
        .position(|a| a == "--lr")
        .and_then(|i| args.get(i + 1)?.parse().ok())
        .unwrap_or(0.003);
    println!("\nTraining for {} epochs, {} steps/epoch, lr={}", num_epochs, steps_per_epoch, lr);
    println!();

    let total_start = Instant::now();
    let warmup_epochs = 2;
    let min_lr = lr * 0.01;

    for epoch in 0..num_epochs {
        let epoch_start = Instant::now();
        let mut epoch_loss = 0.0;

        let current_lr = if epoch < warmup_epochs {
            min_lr + (lr - min_lr) * (epoch + 1) as f32 / warmup_epochs as f32
        } else {
            let progress = (epoch - warmup_epochs) as f32 / (num_epochs - warmup_epochs).max(1) as f32;
            min_lr + 0.5 * (lr - min_lr) * (1.0 + (std::f32::consts::PI * progress).cos())
        };

        for _ in 0..steps_per_epoch {
            let (input_arr, target_arr) = dataset.get_random_batch(batch_size);
            let loss = model.train_step(&input_arr, &target_arr, current_lr);
            epoch_loss += loss;
        }

        let avg_loss = epoch_loss / steps_per_epoch as f32;
        println!("Epoch {:>3} | Loss: {:.4} | LR: {:.6} | Time: {:.2}s",
                 epoch + 1, avg_loss, current_lr, epoch_start.elapsed().as_secs_f64());

        if epoch % 5 == 0 || epoch == num_epochs - 1 {
            generate_sample(&mut model, &tokenizer, 60);
        }
    }

    let total_time = total_start.elapsed().as_secs_f64();
    println!("\nTotal training time: {:.2}s", total_time);
    println!("\n--- Final Generated Text ---");
    generate_sample(&mut model, &tokenizer, 200);
}

fn main() {
    println!("=== Ternary Transformer from Scratch ===\n");
    train();
}
