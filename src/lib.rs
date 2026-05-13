#![recursion_limit = "256"]

pub mod config;
pub mod modules;
pub mod model;
pub mod ternary;
pub mod dataset;

pub use config::AppConfig;
pub use model::TernaryTransformer;
pub use dataset::EncodedDataset;
