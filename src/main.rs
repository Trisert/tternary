#![recursion_limit = "256"]

use tternary::{AppConfig, TernaryTransformer, EncodedDataset};
use burn::prelude::*;
use burn::backend::{Autodiff, NdArray};
use burn::backend::ndarray::NdArrayDevice;
use burn::module::AutodiffModule;
use burn::optim::AdamConfig;
use burn::tensor::{TensorData, activation::softmax};
use burn::train::TrainStep;
use memmap2::Mmap;
use rayon::prelude::*;
use std::io::BufWriter;
use tternary::model::TernaryTransformerTrainingBatch;
use tternary::dataset;
use std::env;
use std::fs::File;
use std::io::Write;
use tokenizers::models::bpe::BPE;
use tokenizers::pre_tokenizers::byte_level::ByteLevel;
use tokenizers::models::bpe::trainer::BpeTrainer;
use tokenizers::tokenizer::{TokenizerImpl, NormalizerWrapper, PreTokenizerWrapper, PostProcessorWrapper, DecoderWrapper};
use tokenizers::Tokenizer;
use ahash::AHashMap;
use rand::Rng;

type MyBackend = Autodiff<NdArray>;
type InnerB = NdArray;

const ENCODED_FILE: &str = "data/tinystories_encoded.bin";
const TOKENIZER_FILE: &str = "data/tokenizer.json";
const BPE_VOCAB_SIZE: usize = 4096;

type BoxedError = Box<dyn std::error::Error + Send + Sync>;

fn train_tokenizer(text_path: &str) -> Result<Tokenizer, BoxedError> {
    println!("Training BPE tokenizer (vocab_size={})...", BPE_VOCAB_SIZE);
    let mut tokenizer: TokenizerImpl<BPE, NormalizerWrapper, PreTokenizerWrapper, PostProcessorWrapper, DecoderWrapper> =
        TokenizerImpl::new(BPE::new(AHashMap::new(), Vec::new()));
    tokenizer.with_pre_tokenizer(Some(ByteLevel::new(false, false, true)));

    let mut trainer = BpeTrainer::builder()
        .vocab_size(BPE_VOCAB_SIZE)
        .build();
    tokenizer.train_from_files(&mut trainer, vec![text_path.to_string()])?;
    println!("  Training complete, saving...");
    tokenizer.save(TOKENIZER_FILE, true)?;
    println!("  Saved to {}", TOKENIZER_FILE);
    let trained = Tokenizer::from_file(TOKENIZER_FILE)?;
    Ok(trained)
}

fn encode_dataset_parallel(tokenizer: &Tokenizer, text_path: &str) -> Result<usize, BoxedError> {
    let file = File::open(text_path)?;
    let mmap = unsafe { Mmap::map(&file)? };
    let text = std::str::from_utf8(&mmap)?;

    println!("  Splitting text into chunks for parallel encoding...");
    let chunk_size = 10_000;
    let lines: Vec<&str> = text.lines().collect();
    let total_lines = lines.len();
    println!("  {} lines to encode", total_lines);

    let out_file = File::create(ENCODED_FILE)?;
    let mut writer = BufWriter::with_capacity(64 * 1024 * 1024, out_file);

    let tokenizer_clone = tokenizer.clone();
    let chunks: Vec<Vec<&str>> = lines.chunks(chunk_size).map(|c| c.to_vec()).collect();

    println!("  Encoding with {} chunks on {} threads...", chunks.len(), rayon::current_num_threads());

    let encoded_chunks: Vec<Vec<Vec<u32>>> = chunks
        .par_iter()
        .map(|chunk| {
            let encodings = tokenizer_clone
                .encode_batch_fast(chunk.clone(), false)
                .unwrap();
            encodings
                .into_iter()
                .map(|e| e.get_ids().to_vec())
                .collect::<Vec<Vec<u32>>>()
        })
        .collect();

    let mut total_tokens = 0usize;
    for chunk in &encoded_chunks {
        for ids in chunk {
            for &id in ids {
                writer.write_all(&id.to_le_bytes())?;
            }
            total_tokens += ids.len();
        }
        if total_tokens % 50_000_000 < chunk_size * 200 {
            print!("\r  Encoded {}M tokens...", total_tokens / 1_000_000);
            std::io::stdout().flush()?;
        }
    }
    writer.flush()?;

    println!("\r  Encoded {}M tokens.      ", total_tokens / 1_000_000);
    Ok(total_tokens)
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

    println!("Tokenizer ready. vocab_size: {}", tokenizer.get_vocab_size(true));

    println!("Encoding dataset (parallel)...");
    let num_tokens = encode_dataset_parallel(&tokenizer, &text_path_str)?;

    Ok((ENCODED_FILE.to_string(), num_tokens, tokenizer))
}

fn generate_sample<B: Backend>(
    model: &TernaryTransformer<B>,
    tokenizer: &Tokenizer,
    num_tokens: usize,
    device: &B::Device,
) {
    let vocab_size = tokenizer.get_vocab_size(true);
    let mut current_tokens: Vec<i32> = vec![2];
    let temperature = 0.8f64;

    for _ in 0..num_tokens {
        let seq = current_tokens.len();
        let input_data = TensorData::new(current_tokens.clone(), [1, seq]);
        let input_tensor = Tensor::<B, 2, Int>::from_data(input_data, device);

        let logits = model.forward_logits(input_tensor);
        let last_logits = logits
            .slice([0..1, seq - 1..seq, 0..vocab_size])
            .reshape([vocab_size]);

        let scaled = last_logits / temperature;
        let probs = softmax(scaled, 0);

        let prob_data = probs.to_data().to_vec::<f32>().unwrap();
        let mut rng = rand::thread_rng();
        let r: f32 = rng.gen_range(0.0f32..1.0);
        let mut cumsum = 0.0f32;
        let mut sampled = (vocab_size - 1) as i32;
        for (i, &p) in prob_data.iter().enumerate() {
            cumsum += p;
            if r <= cumsum {
                sampled = i as i32;
                break;
            }
        }

        current_tokens.push(sampled);
    }

    let ids: Vec<u32> = current_tokens.iter().map(|&t| t as u32).collect();
    let text = tokenizer.decode(&ids, true).unwrap_or_else(|_| "<decode error>".to_string());
    let truncated: String = text.chars().take(500).collect();
    println!("{}", truncated);
}

fn main() {
    println!("=== Ternary Transformer (burn-rs / NdArray) ===\n");

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

    let (encoded_path, num_tokens, tokenizer) = prepare_dataset().expect("Failed to prepare dataset");
    let vocab_size = tokenizer.get_vocab_size(true);
    println!("Vocabulary size: {}", vocab_size);

    let config = AppConfig::new(vocab_size)
        .with_num_epochs(num_epochs)
        .with_steps_per_epoch(steps_per_epoch)
        .with_learning_rate(lr);

    println!("Config: embed_dim={}, hidden={}, layers={}, max_seq={}, kernel={}",
             config.embed_dim, config.hidden_dim,
             config.num_layers, config.max_seq_len, config.kernel_size);

    let device = NdArrayDevice::Cpu;
    let model: TernaryTransformer<MyBackend> = config.init(&device);
    println!("Parameters: {}", model.num_parameters());

    let dataset = EncodedDataset::open(&encoded_path, config.max_seq_len);
    println!("Dataset: {} tokens ({:.1} MB on disk)", num_tokens, num_tokens as f64 * 4.0 / 1e6);

    let mut optim = AdamConfig::new().init::<MyBackend, TernaryTransformer<MyBackend>>();

    println!("\nTraining for {} epochs, {} steps/epoch, lr={}", num_epochs, steps_per_epoch, lr);
    println!();

    let mut model = model;
    let warmup_epochs = 1;
    let min_lr = lr * 0.1;
    let total_start = std::time::Instant::now();

    for epoch in 0..num_epochs {
        let epoch_start = std::time::Instant::now();
        let mut epoch_loss = 0.0f32;

        let current_lr = if epoch < warmup_epochs {
            min_lr + (lr - min_lr) * (epoch + 1) as f64 / warmup_epochs as f64
        } else if num_epochs <= warmup_epochs {
            lr
        } else {
            let progress = (epoch - warmup_epochs) as f64 / (num_epochs - warmup_epochs).max(1) as f64;
            min_lr + 0.5 * (lr - min_lr) * (1.0 + (std::f64::consts::PI * progress).cos())
        };

        for _ in 0..steps_per_epoch {
            let (input_buf, target_buf) = dataset.get_random_batch(config.batch_size);
            let (input_tensor, target_tensor) = dataset::make_batch::<MyBackend>(
                input_buf, target_buf, config.batch_size, config.max_seq_len, &device,
            );

            let batch = TernaryTransformerTrainingBatch {
                inputs: input_tensor,
                targets: target_tensor,
            };

            let output = model.step(batch);
            let loss_val: f32 = output.item.loss.to_data().to_vec::<f32>().unwrap()[0];
            epoch_loss += loss_val;

            model = model.optimize(&mut optim, current_lr, output.grads);
        }

        let avg_loss = epoch_loss / steps_per_epoch as f32;
        println!("Epoch {:>3} | Loss: {:.4} | LR: {:.6} | Time: {:.2}s",
                 epoch + 1, avg_loss, current_lr, epoch_start.elapsed().as_secs_f64());

        if epoch % 5 == 0 || epoch == num_epochs - 1 {
            let inner_model = model.valid();
            generate_sample::<InnerB>(&inner_model, &tokenizer, 60, &device);
        }
    }

    let total_time = total_start.elapsed().as_secs_f64();
    println!("\nTotal training time: {:.2}s", total_time);
    println!("\n--- Final Generated Text ---");
    let inner_model = model.valid();
    generate_sample::<InnerB>(&inner_model, &tokenizer, 200, &device);
}
