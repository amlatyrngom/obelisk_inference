use obelisk::{FunctionalClient, HandlerKit, ScalingState, ServerlessHandler};
use std::{
    sync::Arc,
    time::{Duration, Instant},
};
const USER_MEM: i32 = 512;

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
            Arc::new(FunctionalClient::new("inference", "inferfn", None, Some(USER_MEM)).await);
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
                let (_resp, metadata) = resp.unwrap();
                let end_time = std::time::Instant::now();
                responses.push((end_time.duration_since(start_time), metadata));
                break;
            }
        }
        serde_json::to_string(&responses).unwrap()
    }
}

struct RequestSender {
    curr_avg_latency: f64,
    desired_requests_per_second: f64,
    fc: Arc<FunctionalClient>,
}

impl RequestSender {
    // Next 5 seconds of requests
    async fn send_request_window(&mut self) -> Vec<(Duration, Vec<u8>)> {
        // Window duration.
        let window_duration = 5.0;
        let num_needed_threads = (self.desired_requests_per_second * self.curr_avg_latency).ceil();
        let total_num_requests = window_duration * self.desired_requests_per_second;
        let requests_per_thread = (total_num_requests / num_needed_threads).ceil();
        let actual_total_num_requests = requests_per_thread * num_needed_threads;
        let num_needed_threads = num_needed_threads as u64;
        println!("NT={num_needed_threads}; RPT={requests_per_thread};");
        let mut ts = Vec::new();
        let overall_start_time = Instant::now();
        let prompt = include_str!("prompt.json");
        let req: Vec<String> = serde_json::from_str(prompt).unwrap();
        let payload = serde_json::to_vec(&req).unwrap();
        for _ in 0..num_needed_threads {
            let requests_per_thread = requests_per_thread as u64;
            let fc = self.fc.clone();
            let payload = payload.clone();
            let t = tokio::spawn(async move {
                let start_time = std::time::Instant::now();
                let mut responses = Vec::new();
                let mut curr_idx = 0;
                while curr_idx < requests_per_thread {
                    // Find number of calls to make and update curr idx.
                    let batch_size = 5;
                    let call_count = if requests_per_thread - curr_idx < batch_size {
                        requests_per_thread - curr_idx
                    } else {
                        batch_size
                    };
                    curr_idx += call_count;
                    // Now send requests.
                    let meta = call_count.to_string();
                    let resp = fc.invoke(&meta, &payload).await;
                    if resp.is_err() {
                        println!("Err: {resp:?}");
                        continue;
                    }
                    let (resp, _) = resp.unwrap();
                    let mut resp: Vec<(Duration, Vec<u8>)> = serde_json::from_str(&resp).unwrap();
                    responses.append(&mut resp);
                }
                let end_time = std::time::Instant::now();
                let duration = end_time.duration_since(start_time);

                (duration, responses)
            });
            ts.push(t);
        }
        let mut sum_duration = Duration::from_millis(0);
        let mut all_responses = Vec::new();
        for t in ts {
            let (duration, mut responses) = t.await.unwrap();
            sum_duration = sum_duration.checked_add(duration).unwrap();
            all_responses.append(&mut responses);
        }
        let avg_duration = sum_duration.as_secs_f64() / (actual_total_num_requests);
        self.curr_avg_latency = 0.9 * self.curr_avg_latency + 0.1 * avg_duration;
        let mut avg_req_latency = all_responses.iter().fold(Duration::from_secs(0), |acc, x| {
            acc.checked_add(x.0.clone()).unwrap()
        });
        if all_responses.len() > 0 {
            avg_req_latency = avg_req_latency.div_f64(all_responses.len() as f64);
        }
        println!(
            "AVG_LATENCY={avg_duration}; CURR_AVG_LATENCY={}; REQ_LATENCY={avg_req_latency:?}",
            self.curr_avg_latency
        );
        let overall_end_time = Instant::now();
        let overall_duration = overall_end_time.duration_since(overall_start_time);
        if overall_duration.as_secs_f64() < window_duration {
            let sleep_duration =
                Duration::from_secs_f64(window_duration - overall_duration.as_secs_f64());
            println!("Window sleeping for: {:?}.", sleep_duration);
            tokio::time::sleep(sleep_duration).await;
        }
        all_responses
    }
}

#[cfg(test)]
mod tests {
    use super::RequestSender;
    use super::USER_MEM;
    use obelisk::common::{scaling_state::ScalingStateManager, MetricsManager};
    use obelisk::functional::rescaler::FunctionalScalingInfo;
    use obelisk::FunctionalClient;
    use std::{sync::Arc, time::Duration};

    const STARTING_REQUEST_DURATION_SECS: f64 = 0.200;

    #[tokio::test(flavor = "multi_thread", worker_threads = 16)]
    async fn test_simple_cloud() {
        let fc =
            Arc::new(FunctionalClient::new("inference", "inferfn", None, Some(USER_MEM)).await);
        let prompt = include_str!("prompt.json");
        let req: Vec<String> = serde_json::from_str(prompt).unwrap();
        let meta = String::new();
        let payload = serde_json::to_vec(&req).unwrap();
        let (resp, _) = fc.invoke(&meta, &payload).await.unwrap();
        println!("Infer Resp: {:?}", resp.len());
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 16)]
    async fn test_set_forced_deployment() {
        set_forced_deployment(5.0 * 60.0, 4096, 5, "inference", "inferfn")
            .await
            .unwrap();
    }

    async fn set_forced_deployment(
        duration_secs: f64,
        forced_mem: i32,
        forced_scale: i32,
        namespace: &str,
        name: &str,
    ) -> Result<(), String> {
        let now = chrono::Utc::now();
        let expiry_time = now + chrono::Duration::seconds(duration_secs as i64);
        let forced_deployment = (expiry_time, forced_mem, forced_scale);
        let forced_deployment = bincode::serialize(&forced_deployment).unwrap();
        let mm = MetricsManager::new("functional", namespace, name).await;
        mm.accumulate_metric(forced_deployment).await;
        mm.push_metrics().await.unwrap();
        let sm = ScalingStateManager::new("functional", namespace, name).await;
        sm.start_rescaling_thread().await;
        for _ in 0..10 {
            match sm.retrieve_scaling_state().await {
                Err(x) => {
                    println!("Error: {x:?}. Retrying...");
                    tokio::time::sleep(Duration::from_secs(5)).await;
                    continue;
                }
                Ok(x) => {
                    let scaling_info = x.handler_state.unwrap().scaling_info;
                    if scaling_info.is_none() {
                        println!("Scaling info not set. Retrying...");
                        tokio::time::sleep(Duration::from_secs(5)).await;
                        continue;
                    }
                    let scaling_info = scaling_info.unwrap();
                    let scaling_info: FunctionalScalingInfo =
                        serde_json::from_str(&scaling_info).unwrap();
                    let forced_deployment = scaling_info.forced_deployment;
                    if forced_deployment.is_none() {
                        println!("Forced deployment not set. Retrying...");
                        tokio::time::sleep(Duration::from_secs(5)).await;
                        continue;
                    }
                    let forced_deployment = forced_deployment.unwrap();
                    if forced_deployment.0 != expiry_time {
                        println!("Forced deployment corret expiry time not set. Retrying...");
                        tokio::time::sleep(Duration::from_secs(5)).await;
                        continue;
                    }
                    println!("Forced deployment set: {:?}.", forced_deployment);
                    return Ok(());
                }
            }
        }
        Err("Could not set forced deployment.".to_string())
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 16)]
    async fn test_scaling_cloud() {
        let fc =
            Arc::new(FunctionalClient::new("inference", "inferfn", None, Some(USER_MEM)).await);
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

    // #[tokio::test(flavor = "multi_thread", worker_threads = 16)]
    // async fn test_bench_cloud() {
    //     let fc =
    //         Arc::new(FunctionalClient::new("inference", "benchfn", None, Some(USER_MEM)).await);
    //     let req = ("Amadou is boss.".to_string(), "Who is boss?".to_string());
    //     let meta = String::new();
    //     let payload = serde_json::to_vec(&req).unwrap();
    //     let (resp, _) = fc.invoke(&meta, &payload).await.unwrap();
    //     let resp: Vec<(Duration, String)> = serde_json::from_str(&resp).unwrap();
    //     println!("Infer Resp: {resp:?}");
    // }

    /// Write bench output.
    async fn write_bench_output(points: Vec<(u64, f64, Vec<u8>)>, expt_name: &str) {
        let expt_dir = "results/infer_bench";
        std::fs::create_dir_all(expt_dir).unwrap();
        let mut writer = csv::WriterBuilder::new()
            .from_path(format!("{expt_dir}/{expt_name}.csv"))
            .unwrap();
        for (since, duration, metadata) in points {
            let (mem, is_lambda): (i32, bool) = serde_json::from_slice(&metadata).unwrap();
            let mode = if is_lambda { "Lambda" } else { "ECS" };
            writer
                .write_record(&[
                    since.to_string(),
                    duration.to_string(),
                    mem.to_string(),
                    mode.to_string(),
                ])
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

    async fn run_bench(resquest_sender: &mut RequestSender, name: &str, test_duration: Duration) {
        let mut results: Vec<(u64, f64, Vec<u8>)> = Vec::new();
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
            let resp: Vec<(Duration, Vec<u8>)> = resquest_sender.send_request_window().await;
            for (duration, metadata) in &resp {
                results.push((since, duration.as_secs_f64(), metadata.clone()));
            }
        }
        write_bench_output(results, &format!("{name}")).await;
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 16)]
    async fn full_bench_cloud() {
        let set_force = true;
        if set_force {
            // Time upper bound: 1 hour 10 minutes.
            let time_upper_bound = 60.0 * 60.0 + 10.0 * 60.0;
            let forced_mem = 4096;
            let forced_scale = 4;
            set_forced_deployment(time_upper_bound, forced_mem, forced_scale, "inference", "benchfn")
                .await
                .unwrap();

        }

        let fc =
            Arc::new(FunctionalClient::new("inference", "benchfn", None, Some(USER_MEM)).await);
        let duration_mins = 15.0;
        let low_req_per_secs = 1.0;
        let medium_req_per_secs = 10.0;
        let high_req_per_secs = 100.0;
        let mut request_sender = RequestSender {
            curr_avg_latency: STARTING_REQUEST_DURATION_SECS,
            desired_requests_per_second: 0.0,
            fc: fc.clone(),
        };
        // Low
        request_sender.desired_requests_per_second = low_req_per_secs;
        run_bench(
            &mut request_sender,
            "pre_low",
            Duration::from_secs_f64(60.0 * duration_mins),
        )
        .await;
        // // Medium
        // request_sender.desired_requests_per_second = medium_req_per_secs;
        // run_bench(
        //     &mut request_sender,
        //     "pre_medium",
        //     Duration::from_secs_f64(60.0 * duration_mins),
        // )
        // .await;
        // // High
        // request_sender.desired_requests_per_second = high_req_per_secs;
        // run_bench(
        //     &mut request_sender,
        //     "pre_high",
        //     Duration::from_secs_f64(60.0 * duration_mins),
        // )
        // .await;
        // // Low again.
        // request_sender.desired_requests_per_second = low_req_per_secs;
        // run_bench(
        //     &mut request_sender,
        //     "post_low",
        //     Duration::from_secs_f64(60.0 * duration_mins),
        // )
        // .await;
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 16)]
    async fn full_dummy_bench_cloud() {
        let fc =
            Arc::new(FunctionalClient::new("inference", "benchfn", None, Some(USER_MEM)).await);
        // Low Mode
        let desired_requests_per_second = 400.0;
        let curr_avg_latency = STARTING_REQUEST_DURATION_SECS;
        let mut request_sender = RequestSender {
            curr_avg_latency,
            desired_requests_per_second,
            fc: fc.clone(),
        };
        for _ in 0..10 {
            let responses = request_sender.send_request_window().await;
            println!("Num Responses: {}", responses.len());
        }
    }
}
