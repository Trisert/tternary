use burn::prelude::*;
use burn::tensor::TensorData;
use memmap2::Mmap;
use rand::Rng;
use std::fs::File;

pub struct EncodedDataset {
    mmap: Mmap,
    len: usize,
    max_seq_len: usize,
}

unsafe impl Send for EncodedDataset {}
unsafe impl Sync for EncodedDataset {}

impl EncodedDataset {
    pub fn open(path: &str, max_seq_len: usize) -> Self {
        let file = File::open(path).expect("Failed to open encoded dataset");
        let len = file.metadata().expect("Failed to get file metadata").len() as usize / 2;
        let mmap = unsafe { Mmap::map(&file).expect("Failed to mmap dataset") };
        Self { mmap, len, max_seq_len }
    }

    #[inline]
    fn token_at(&self, idx: usize) -> u32 {
        let byte_off = idx * 2;
        u16::from_le_bytes([
            self.mmap[byte_off],
            self.mmap[byte_off + 1],
        ]) as u32
    }

    pub fn get_random_batch(&self, batch_size: usize) -> (Vec<i32>, Vec<i32>) {
        let mut rng = rand::thread_rng();
        let max_start = self.len.saturating_sub(self.max_seq_len + 1);
        let seq = self.max_seq_len;
        let total = batch_size * seq;

        let mut input_buf = Vec::<i32>::with_capacity(total);
        let mut target_buf = Vec::<i32>::with_capacity(total);

        for _ in 0..batch_size {
            let start = rng.gen_range(0..max_start);
            for j in 0..seq {
                input_buf.push(self.token_at(start + j) as i32);
                target_buf.push(self.token_at(start + j + 1) as i32);
            }
        }

        (input_buf, target_buf)
    }

    pub fn len(&self) -> usize {
        self.len
    }
}

pub fn make_batch<B: Backend>(
    inputs: Vec<i32>,
    targets: Vec<i32>,
    batch_size: usize,
    seq_len: usize,
    device: &B::Device,
) -> (Tensor<B, 2, Int>, Tensor<B, 2, Int>) {
    let input_tensor = Tensor::from_data(
        TensorData::new(inputs, [batch_size, seq_len]),
        device,
    );
    let target_tensor = Tensor::from_data(
        TensorData::new(targets, [batch_size, seq_len]),
        device,
    );
    (input_tensor, target_tensor)
}
