use rust_bert::pipelines::sentence_embeddings::{
    SentenceEmbeddingsBuilder, SentenceEmbeddingsModelType,
};
use rust_bert::pipelines::translation::{Language, TranslationModelBuilder};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // TODO: Put behind env var.
    // tokio::task::block_in_place(move || {
    //     // Download and cache model.
    //     // std::env::set_var("RUSTBERT_CACHE", obelisk::common::local_storage_path());
    //     // let translation_model = TranslationModelBuilder::new()
    //     //     .with_source_languages(vec![Language::English])
    //     //     .with_target_languages(vec![Language::French])
    //     //     .create_model().unwrap();
    //     // let _qa_model = QuestionAnsweringModel::new(Default::default()).unwrap();

    //     let model = SentenceEmbeddingsBuilder::remote(
    //         SentenceEmbeddingsModelType::AllMiniLmL12V2
    //     ).create_model().unwrap();

    //     let sentences = [
    //         "this is an example sentence",
    //     ];
    //     let start_time = std::time::Instant::now();
    //     let _output = model.encode(&sentences);
    //     let end_time = std::time::Instant::now();
    //     let duration = end_time.duration_since(start_time);
    //     println!("Duration: {duration:?}");
    //     let start_time = std::time::Instant::now();
    //     let _output = model.encode(&sentences);
    //     let end_time = std::time::Instant::now();
    //     let duration = end_time.duration_since(start_time);
    //     println!("Duration: {duration:?}");
    //     let start_time = std::time::Instant::now();
    //     let _output = model.encode(&sentences);
    //     let end_time = std::time::Instant::now();
    //     let duration = end_time.duration_since(start_time);
    //     println!("Duration: {duration:?}");
    // });

    // TODO: Replace with "deployment.toml" when done testing.
    let main_deployment = include_str!("deployment.toml");
    let deployments: Vec<String> = vec![main_deployment.into()];
    obelisk_deployment::build_user_deployment("inference", &deployments).await;

    Ok(())
}
