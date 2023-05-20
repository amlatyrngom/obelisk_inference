use obelisk::{FunctionInstance, FunctionalClient};
use serde_json::Value;
use std::sync::Arc;

pub struct BenchFn {
    infer_client: Arc<FunctionalClient>,
}

#[async_trait::async_trait]
impl FunctionInstance for BenchFn {
    /// Invocation.
    async fn invoke(&self, arg: Value) -> Value {
        self.handle_req(arg).await
    }
}

impl BenchFn {
    /// Create.
    pub async fn new() -> Self {
        let infer_client = Arc::new(FunctionalClient::new("inference").await);
        BenchFn { infer_client }
    }

    /// Handle requests.
    pub async fn handle_req(&self, req: Value) -> Value {
        let mut responses = Vec::new();
        for _ in 0..3 {
            let req = req.to_string();
            let req = req.as_bytes();
            loop {
                let start_time = std::time::Instant::now();
                let resp = self.infer_client.invoke_internal(req).await;
                if resp.is_err() {
                    continue;
                }
                let resp = resp.unwrap();
                let end_time = std::time::Instant::now();
                responses.push((end_time.duration_since(start_time), resp));
                break;
            }
        }
        serde_json::to_value(responses).unwrap()
    }
}
