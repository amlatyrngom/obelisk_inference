use obelisk::{FunctionalClient, HandlerKit, ScalingState, ServerlessHandler};
use std::sync::Arc;

pub struct BenchFn {
    infer_client: Arc<FunctionalClient>,
}

#[async_trait::async_trait]
impl ServerlessHandler for BenchFn {
    /// Invocation.
    async fn handle(&self, meta: String, payload: Vec<u8>) -> (String, Vec<u8>) {
        (self.handle_req(meta, payload).await, vec![])
    }

    async fn checkpoint(&self, _scaling_state: &ScalingState, _terminating: bool) {}
}

impl BenchFn {
    /// Create.
    pub async fn new(_kit: HandlerKit) -> Self {
        let infer_client =
            Arc::new(FunctionalClient::new("inference", "inferfn", None, Some(512)).await);
        BenchFn { infer_client }
    }

    /// Handle requests.
    pub async fn handle_req(&self, meta: String, payload: Vec<u8>) -> String {
        let mut responses = Vec::new();
        let calls_per_round: i32 = meta.parse().unwrap();
        let meta = String::new();
        for _ in 0..calls_per_round {
            loop {
                let start_time = std::time::Instant::now();
                let resp = self.infer_client.invoke(&meta, &payload).await;
                if resp.is_err() {
                    continue;
                }
                let (resp, _) = resp.unwrap();
                let end_time = std::time::Instant::now();
                responses.push((end_time.duration_since(start_time), resp));
                break;
            }
        }
        serde_json::to_string(&responses).unwrap()
    }
}

#[cfg(test)]
mod tests {
    use obelisk::FunctionalClient;
    use std::{sync::Arc, time::Duration};

    #[tokio::test(flavor = "multi_thread", worker_threads = 16)]
    async fn test_simple_cloud() {
        let fc = Arc::new(FunctionalClient::new("inference", "inferfn", None, Some(512)).await);
        let req = ("Amadou is boss.".to_string(), "Who is boss?".to_string());
        let meta = String::new();
        let payload = serde_json::to_vec(&req).unwrap();
        let (resp, _) = fc.invoke(&meta, &payload).await.unwrap();
        println!("Infer Resp: {resp:?}");
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 16)]
    async fn test_scaling_cloud() {
        let fc = Arc::new(FunctionalClient::new("inference", "inferfn", None, Some(512)).await);
        let req = ("Amadou is boss.".to_string(), "Who is boss?".to_string());
        let meta = String::new();
        let payload = serde_json::to_vec(&req).unwrap();
        let num_threads = 2;
        let mut ts = Vec::new();
        for i in 0..num_threads {
            let meta = meta.clone();
            let payload = payload.clone();
            let fc = fc.clone();
            ts.push(tokio::spawn(async move {
                for _ in 0..2000 {
                    let (resp, _) = fc.invoke(&meta, &payload).await.unwrap();
                    println!("Infer Resp {i}: {resp:?}");
                }
            }));
        }

        for t in ts {
            let _ = t.await;
        }
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 16)]
    async fn test_bench_cloud() {
        let fc = Arc::new(FunctionalClient::new("inference", "benchfn", None, Some(512)).await);
        let req = ("Amadou is boss.".to_string(), "Who is boss?".to_string());
        let meta = String::new();
        let payload = serde_json::to_vec(&req).unwrap();
        let (resp, _) = fc.invoke(&meta, &payload).await.unwrap();
        let resp: Vec<(Duration, String)> = serde_json::from_str(&resp).unwrap();
        println!("Infer Resp: {resp:?}");
    }

    /// Write bench output.
    async fn write_bench_output(points: Vec<(u64, f64, String)>, expt_name: &str) {
        let expt_dir = "results/infer_bench";
        std::fs::create_dir_all(expt_dir).unwrap();
        let mut writer = csv::WriterBuilder::new()
            .from_path(format!("{expt_dir}/{expt_name}.csv"))
            .unwrap();
        for (since, duration, mode) in points {
            writer
                .write_record(&[since.to_string(), duration.to_string(), mode])
                .unwrap();
        }
        writer.flush().unwrap();
    }

    #[derive(Debug)]
    enum RequestRate {
        Low,
        Medium,
        High(usize),
    }

    async fn run_bench(
        fc: Arc<FunctionalClient>,
        rate: RequestRate,
        prefix: &str,
        test_duration: Duration,
    ) {
        let (activity, num_workers, calls_per_round) = match rate {
            RequestRate::Low => (0.1, 1, 5),
            RequestRate::Medium => (1.0, 1, 20),
            RequestRate::High(num_workers) => (1.0, num_workers, 20),
        };
        let mut workers = Vec::new();
        let req = ("Amadou is boss.".to_string(), "Who is boss?".to_string());
        let meta = calls_per_round.to_string();
        let payload = serde_json::to_vec(&req).unwrap();
        for n in 0..num_workers {
            let fc = fc.clone();
            let test_duration = test_duration.clone();
            let meta = meta.clone();
            let payload = payload.clone();
            workers.push(tokio::spawn(async move {
                let mut results: Vec<(u64, f64, String)> = Vec::new();
                let start_time = std::time::Instant::now();
                loop {
                    // Pick an image at random.
                    // TODO: Find a better way to select images.
                    let curr_time = std::time::Instant::now();
                    let since = curr_time.duration_since(start_time);
                    if since > test_duration {
                        break;
                    }
                    let since = since.as_millis() as u64;
                    let resp = fc.invoke(&meta, &payload).await;
                    if resp.is_err() {
                        println!("Err: {resp:?}");
                        continue;
                    }
                    let (resp, _) = resp.unwrap();
                    let resp: Vec<(Duration, String)> = serde_json::from_str(&resp).unwrap();
                    if n < 2 {
                        println!("Worker {n}. Resp: {resp:?}.");
                    }
                    let infer_times: Vec<_> =
                        resp.into_iter().map(|(x, _y)| x.as_secs_f64()).collect();
                    if infer_times.is_empty() {
                        continue;
                    }
                    for duration in &infer_times {
                        results.push((since, *duration, "Infer".into()));
                    }
                    // Simulate lambda.
                    let end_time = std::time::Instant::now();
                    let mut active_time_ms =
                        end_time.duration_since(curr_time).as_secs_f64() * 1000.0;
                    active_time_ms += 50.0 * infer_times.len() as f64;
                    println!("Active Time MS: {active_time_ms}.");
                    // Decide wait time.
                    let mut wait_time_ms = active_time_ms / activity - active_time_ms;
                    if wait_time_ms > 30.0 * 1000.0 {
                        wait_time_ms = 30.0 * 1000.0; // Prevent excessive waiting.
                    }
                    if wait_time_ms > 1.0 {
                        let wait_time =
                            std::time::Duration::from_millis(wait_time_ms.ceil() as u64);
                        println!("Waiting {wait_time:?}.");
                        tokio::time::sleep(wait_time).await;
                    }
                }
                results
            }));
        }
        let mut results = Vec::new();
        for w in workers {
            let mut r = w.await.unwrap();
            results.append(&mut r);
        }
        write_bench_output(results, &format!("{prefix}_{rate:?}")).await;
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 16)]
    async fn full_bench_cloud() {
        let fc = Arc::new(FunctionalClient::new("inference", "benchfn", None, Some(512)).await);
        let duration_mins = 30.0;
        run_bench(
            fc.clone(),
            RequestRate::Low,
            "pre",
            Duration::from_secs_f64(60.0 * duration_mins),
        )
        .await;
        run_bench(
            fc.clone(),
            RequestRate::Medium,
            "pre",
            Duration::from_secs_f64(60.0 * duration_mins),
        )
        .await;
        run_bench(
            fc.clone(),
            RequestRate::High(10),
            "pre",
            Duration::from_secs_f64(60.0 * duration_mins),
        )
        .await;
        run_bench(
            fc.clone(),
            RequestRate::Low,
            "post",
            Duration::from_secs_f64(60.0 * duration_mins),
        )
        .await;
    }
}
