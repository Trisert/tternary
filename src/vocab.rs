use std::collections::HashMap;

pub struct Vocabulary {
    pub token_to_id: HashMap<String, usize>,
    pub id_to_token: HashMap<usize, String>,
    pub vocab_size: usize,
}

impl Vocabulary {
    pub fn new() -> Self {
        Self {
            token_to_id: HashMap::new(),
            id_to_token: HashMap::new(),
            vocab_size: 0,
        }
    }

    pub fn build_from_text(text: &str, max_vocab_size: usize) -> Self {
        let mut vocab = Vocabulary::new();

        let mut char_counts: HashMap<char, usize> = HashMap::new();
        for c in text.chars() {
            *char_counts.entry(c).or_insert(0) += 1;
        }

        let mut chars: Vec<(char, usize)> = char_counts.into_iter().collect();
        chars.sort_by(|a, b| b.1.cmp(&a.1));

        let special_tokens = vec!["<PAD>", "<UNK>"];
        for token in special_tokens {
            vocab.token_to_id.insert(token.to_string(), vocab.vocab_size);
            vocab.id_to_token.insert(vocab.vocab_size, token.to_string());
            vocab.vocab_size += 1;
        }

        for (c, _) in chars.into_iter().take(max_vocab_size - 2) {
            let token = c.to_string();
            vocab.token_to_id.insert(token.clone(), vocab.vocab_size);
            vocab.id_to_token.insert(vocab.vocab_size, token);
            vocab.vocab_size += 1;
        }

        vocab
    }

    pub fn encode_char(&self, c: char) -> u16 {
        let token = c.to_string();
        self.token_to_id.get(&token).copied().unwrap_or(1) as u16
    }

    pub fn encode(&self, text: &str) -> Vec<u16> {
        text.chars().map(|c| self.encode_char(c)).collect()
    }

    pub fn decode(&self, ids: &[usize]) -> String {
        ids.iter()
            .map(|&id| self.id_to_token.get(&id).cloned().unwrap_or_else(|| "<UNK>".to_string()))
            .collect()
    }
}

impl Default for Vocabulary {
    fn default() -> Self {
        Self::new()
    }
}
