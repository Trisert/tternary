use ndarray::Array2;
use rand::Rng;
use std::fs::File;
use std::io::{Read, Seek, SeekFrom};

pub struct Dataset {
    file: File,
    len: usize,
    max_seq_len: usize,
}

impl Dataset {
    pub fn open(path: &str, max_seq_len: usize) -> Self {
        let file = File::open(path).expect("Failed to open encoded dataset");
        let len = file.metadata().expect("Failed to get file metadata").len() as usize / 4;
        Self { file, len, max_seq_len }
    }

    fn read_tokens(&mut self, start: usize, count: usize) -> Vec<u32> {
        let mut buf = vec![0u8; count * 4];
        self.file.seek(SeekFrom::Start(start as u64 * 4)).unwrap();
        self.file.read_exact(&mut buf).unwrap();
        buf.chunks_exact(4)
            .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect()
    }

    pub fn get_random_batch(&mut self, batch_size: usize) -> (Array2<f32>, Array2<f32>) {
        let mut rng = rand::thread_rng();
        let max_start = self.len.saturating_sub(self.max_seq_len + 1);
        let mut input = Array2::zeros((batch_size, self.max_seq_len));
        let mut target = Array2::zeros((batch_size, self.max_seq_len));

        for i in 0..batch_size {
            let start = rng.gen_range(0..max_start);
            let tokens = self.read_tokens(start, self.max_seq_len + 1);
            for j in 0..self.max_seq_len {
                input[[i, j]] = tokens[j] as f32;
                target[[i, j]] = tokens[j + 1] as f32;
            }
        }
        (input, target)
    }

    pub fn len(&self) -> usize {
        self.len
    }
}
