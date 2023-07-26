// use rust_bert::pipelines::translation::{TranslationModelBuilder, Language};
use rust_bert::pipelines::question_answering::QuestionAnsweringModel;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // TODO: Put behind env var.
    // tokio::task::block_in_place(move || {
    //     // Download and cache model.
    //     std::env::set_var("RUSTBERT_CACHE", obelisk::common::local_storage_path());
    //     // let translation_model = TranslationModelBuilder::new()
    //     //     .with_source_languages(vec![Language::English])
    //     //     .with_target_languages(vec![Language::French])
    //     //     .create_model().unwrap();
    //     let _qa_model = QuestionAnsweringModel::new(Default::default()).unwrap();
    // });

    // TODO: Replace with "deployment.toml" when done testing.
    let main_deployment = include_str!("deployment.toml");
    let deployments: Vec<String> = vec![main_deployment.into()];
    obelisk_deployment::build_user_deployment("inference", &deployments).await;

    Ok(())
}
