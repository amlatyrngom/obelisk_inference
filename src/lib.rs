// use rust_bert::pipelines::common::ModelResource;
use rust_bert::pipelines::question_answering::{
    QaInput, QuestionAnsweringConfig, QuestionAnsweringModel,
};
use rust_bert::pipelines::sentence_embeddings::{
    SentenceEmbeddingsBuilder, SentenceEmbeddingsModel, SentenceEmbeddingsModelType,
};

// use rust_bert::resources::{LocalResource, RemoteResource, ResourceProvider};
// use std::path::PathBuf;
use std::time::Instant;
// use rust_bert::pipelines::translation::{TranslationModelBuilder, Language};
use obelisk::{HandlerKit, ScalingState, ServerlessHandler};
use serde::{Deserialize, Serialize};
// use std::any::Any;
use std::sync::Arc;
use tokio::runtime::Handle;
use tokio::sync::{mpsc, oneshot, Mutex};
mod bench_fn;
pub use bench_fn::BenchFn;

#[derive(Debug)]
struct InferCmd {
    input: (String, String),
    resp_tx: oneshot::Sender<String>,
}

#[derive(Serialize, Deserialize, Clone, Debug)]
struct InferResp {
    answer: String,
}

#[derive(Clone)]
pub struct InferFn {
    tx: mpsc::Sender<InferCmd>,
    metadata: Vec<u8>,
    model: Arc<Mutex<SentenceEmbeddingsModel>>,
}

#[async_trait::async_trait]
impl ServerlessHandler for InferFn {
    async fn handle(&self, _meta: String, payload: Vec<u8>) -> (String, Vec<u8>) {
        let input = serde_json::from_slice(&payload).unwrap();
        self.handle_req(input).await
    }

    async fn checkpoint(&self, _scaling_state: &ScalingState, _terminating: bool) {}
}

impl InferFn {
    /// Create.
    pub async fn new(kit: HandlerKit) -> Self {
        let (tx, rx) = mpsc::channel(32);
        // Spawn enough tasks.
        // let num_threads = (kit.instance_info.cpus as f64 / 1024.0).ceil() as i32;
        // println!("Spawning {num_threads:?} threads.");
        // for _ in 0..num_threads {
        //     let handle = Handle::current();
        //     let rx = rx.clone();
        //     tokio::task::spawn_blocking(move || {
        //         let _ = Self::run_inferer(rx, done_tx, handle);
        //     });
        //     let _ = done_rx.await;
        // }
        // let handle = Handle::current();
        // let (done_tx, done_rx) = oneshot::channel();
        // tokio::task::spawn_blocking(move || {
        //     let _ = Self::run_inferer(rx, done_tx, handle);
        // });
        // let _ = done_rx.await;
        let metadata = (
            kit.instance_info.mem,
            kit.instance_info.private_url.is_none(),
        );
        let metadata = serde_json::to_vec(&metadata).unwrap();
        // Make model.
        let local_data_dir = obelisk::common::local_storage_path();
        let cache_dir = format!("{local_data_dir}/rustbert");
        std::env::set_var("RUSTBERT_CACHE", cache_dir);
        println!("Making default config!");
        let model = tokio::task::block_in_place(|| {
            let model =
                SentenceEmbeddingsBuilder::remote(SentenceEmbeddingsModelType::AllMiniLmL12V2)
                    .create_model()
                    .unwrap();
            Arc::new(Mutex::new(model))
        });
        InferFn {
            tx,
            metadata,
            model,
        }
    }

    /// Handle request.
    async fn handle_req(&self, inputs: Vec<String>) -> (String, Vec<u8>) {
        // let (tx, rx) = oneshot::channel();
        println!("Request Received. Input length: {}", inputs.len());
        // let cmd = InferCmd { resp_tx: tx, input };
        // Hacky fix.
        // let res = self.tx.send(cmd).await;
        let start_time = std::time::Instant::now();
        let answer = {
            let model = self.model.lock().await;
            let resp = tokio::task::block_in_place(|| model.encode(&inputs).unwrap_or(Vec::new()));
            resp.first().cloned().unwrap_or(Vec::new())
        };
        let end_time = std::time::Instant::now();
        let answer = serde_json::to_string(&answer).unwrap();
        let duration = end_time.duration_since(start_time);
        println!("Inference duration: {duration:?}");
        (answer, self.metadata.clone())
    }

    // fn get_bert_resource(subdir: &str) -> PathBuf {
    //     let local_data_dir = obelisk::common::local_storage_path();
    //     let dir = format!("{local_data_dir}/distilbert-qa/{subdir}");
    //     let dir = std::fs::read_dir(dir).unwrap();
    //     for entry in dir {
    //         let entry = entry.unwrap();
    //         let filename = entry.file_name().to_str().unwrap().to_string();
    //         if !filename.ends_with(".meta") && !filename.ends_with(".lock") {
    //             return entry.path();
    //         }
    //     }
    //     panic!("Cannot find resource");
    // }

    // /// Run inferer.
    // fn run_inferer(
    //     rx: mpsc::Receiver<InferCmd>,
    //     done_tx: oneshot::Sender<()>,
    //     handle: Handle,
    // ) -> Result<(), String> {
    //     let mut rx = rx;
    //     let local_data_dir = obelisk::common::local_storage_path();
    //     let cache_dir = format!("{local_data_dir}/rustbert");
    //     std::env::set_var("RUSTBERT_CACHE", cache_dir);
    //     println!("Making default config!");
    //     // It is faster to just download the models.
    //     // So just use the default config.
    //     // let config = QuestionAnsweringConfig::default();
    //     // config.model_resource = ModelResource::Torch(Box::new(LocalResource {
    //     //     local_path: Self::get_bert_resource("model"),
    //     // }));
    //     // config.vocab_resource = Box::new(LocalResource {
    //     //     local_path: Self::get_bert_resource("vocab"),
    //     // });
    //     // config.config_resource = Box::new(LocalResource {
    //     //     local_path: Self::get_bert_resource("config"),
    //     // });
    //     // let translation_model = TranslationModelBuilder::new()
    //     //     .with_source_languages(vec![Language::English])
    //     //     .with_target_languages(vec![Language::French])
    //     //     .create_model().unwrap();
    //     let embedding_model = SentenceEmbeddingsBuilder::remote(
    //         SentenceEmbeddingsModelType::AllMiniLmL12V2
    //     ).create_model().unwrap();

    //     // let qa_model = QuestionAnsweringModel::new(config).unwrap();
    //     let _ = done_tx.send(());
    //     loop {
    //         // let start_time = std::time::Instant::now();
    //         let mut inputs = Vec::new();
    //         let mut txs = Vec::new();
    //         println!("Inferer loop running!");
    //         // Read first cmd.
    //         let cmd = handle.block_on(rx.recv());
    //         let cmd = match cmd {
    //             None => {
    //                 println!("Inferer Stopped!");
    //                 break;
    //             }
    //             Some(cmd) => cmd,
    //         };
    //         let start_time = Instant::now();
    //         // inputs.push(QaInput {
    //         //     context: cmd.input.0,
    //         //     question: cmd.input.1,
    //         // });
    //         inputs.push(cmd.input.0);
    //         txs.push(cmd.resp_tx);
    //         // // Read all pending commands into a batch.
    //         // loop {
    //         //     let cmd = rx.try_recv();
    //         //     if let Ok(cmd) = cmd {
    //         //         // inputs.push(QaInput {
    //         //         //     context: cmd.input.0,
    //         //         //     question: cmd.input.1,
    //         //         // });
    //         //         inputs.push(cmd.input.0);
    //         //         txs.push(cmd.resp_tx);
    //         //     } else {
    //         //         break;
    //         //     }
    //         // }

    //         // let end_time = std::time::Instant::now();
    //         // let duration = end_time.duration_since(start_time);
    //         // println!("Read Inputs. Duration: {duration:?}");
    //         // let start_time = std::time::Instant::now();
    //         // let results = translation_model
    //         //     .translate(&texts, Language::English, Language::French)
    //         //     .unwrap();
    //         let results = qa_model
    //             .predict(&inputs, 1, 32)
    //             .into_iter()
    //             .map(|answers| answers.first().map_or(String::new(), |a| a.answer.clone()))
    //             .collect::<Vec<_>>();
    //         println!("Inferer loop found: {results:?}!");
    //         // let end_time = std::time::Instant::now();
    //         // let duration = end_time.duration_since(start_time);
    //         // println!("Run code. Duration: {duration:?}");
    //         // let start_time = std::time::Instant::now();
    //         for (tx, res) in txs.into_iter().zip(results) {
    //             let _ = tx.send(res);
    //         }
    //         let end_time = std::time::Instant::now();
    //         let duration = end_time.duration_since(start_time);
    //         println!("Request-Respond. Duration: {duration:?}");
    //     }
    //     println!("Exiting inferer!");
    //     return Ok(());
    // }
}

#[cfg(test)]
mod tests {
    use super::InferFn;
    use obelisk::{HandlerKit, InstanceInfo};
    use std::sync::Arc;

    fn dummy_kit() -> HandlerKit {
        let instance_info = InstanceInfo {
            peer_id: "qwe".into(),
            mem: 512,
            cpus: 256,
            az: None,
            public_url: None,
            private_url: None,
            service_name: None,
            handler_name: None,
            subsystem: "bla".into(),
            namespace: "bla".into(),
            identifier: "bla".into(),
            unique: false,
            persistent: false,
        };

        HandlerKit {
            instance_info: Arc::new(instance_info),
            serverless_storage: None,
        }
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 16)]
    async fn test_infer_fn() {
        let kit = dummy_kit();
        let inferfn = InferFn::new(kit).await;
        // let context = "This text, from Amadou, is not very hard to translate.".to_string();
        // let question = "Who is the text from?".to_string();
        // let input = (context, question);
        let inputs = include_str!("prompt.json");
        let inputs: Vec<String> = serde_json::from_str(inputs).unwrap();
        let (answer, _) = inferfn.handle_req(inputs.clone()).await;
        println!("Answer: {}.", answer.len());
        let (answer, _) = inferfn.handle_req(inputs.clone()).await;
        println!("Answer: {}.", answer.len());
        // inferfn.tx;
    }
}
