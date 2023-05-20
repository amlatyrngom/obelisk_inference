#[cfg(test)]
mod tests {
    use crate::{InferFn, InferReq, InferResp};
    use base64::{engine::general_purpose, Engine as _};
    use image::io::Reader as ImageReader;
    use obelisk::FunctionalClient;
    use serde_json::Value;
    use std::sync::Arc;

    #[tokio::test(flavor = "multi_thread", worker_threads = 16)]
    async fn test_simple_local() {
        let mut reqs = Vec::new();
        for filename in ["panda", "lion", "elephant"] {
            let img: image::DynamicImage = ImageReader::open(format!("{filename}.jpeg"))
                .unwrap()
                .decode()
                .unwrap();
            let req = InferReq {
                width: img.width(),
                height: img.height(),
                img: general_purpose::STANDARD_NO_PAD.encode(img.into_bytes()),
            };
            reqs.push(req);
        }
        let infer_fn = InferFn::new().await;
        let num_threads = 8;
        let mut ts = Vec::new();
        for i in 0..num_threads {
            let infer_fn = infer_fn.clone();
            let req = reqs[i % reqs.len()].clone();
            ts.push(tokio::spawn(async move {
                // Slow one down to test parallelism.
                if i == 0 {
                    tokio::time::sleep(std::time::Duration::from_secs(5)).await;
                }
                let start_time = std::time::Instant::now();
                let resp = infer_fn.handle_request(req).await;
                let end_time = std::time::Instant::now();
                let duration = end_time.duration_since(start_time);
                println!("Infer Resp: {resp:?}. Duration: {duration:?}");
            }));
        }
        for t in ts {
            t.await.unwrap();
        }
        // Repeat to avoid initialization effects.
        let mut ts = Vec::new();
        for i in 0..num_threads {
            let infer_fn = infer_fn.clone();
            let req = reqs[i % reqs.len()].clone();
            ts.push(tokio::spawn(async move {
                let start_time = std::time::Instant::now();
                let resp = infer_fn.handle_request(req).await;
                let end_time = std::time::Instant::now();
                let duration = end_time.duration_since(start_time);
                println!("Infer Resp: {resp:?}. Duration: {duration:?}");
            }));
        }
        for t in ts {
            t.await.unwrap();
        }
    }

    #[tokio::test]
    async fn test_simple_cloud() {
        let fc = Arc::new(FunctionalClient::new("inference").await);
        let img = ImageReader::open("panda.jpeg").unwrap().decode().unwrap();
        let num_threads = 2;
        let mut ts = Vec::new();
        let req = InferReq {
            width: img.width(),
            height: img.height(),
            img: general_purpose::STANDARD_NO_PAD.encode(img.into_bytes()),
        };
        for _ in 0..num_threads {
            let fc = fc.clone();
            let req = req.clone();
            ts.push(tokio::spawn(async move {
                let req = serde_json::to_vec(&req).unwrap();
                let resp = fc.invoke(&req).await.unwrap();
                let resp: InferResp = serde_json::from_value(resp).unwrap();
                println!("Infer Resp: {resp:?}");
            }));
        }
        for t in ts {
            t.await.unwrap();
        }
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

    async fn run_bench(fc: Arc<FunctionalClient>, rate: RequestRate, prefix: &str) {
        let (activity, num_workers) = match rate {
            RequestRate::Low => (0.1, 1),
            RequestRate::Medium => (1.0, 2),
            RequestRate::High(num_workers) => (1.0, num_workers),
        };
        let test_duration = std::time::Duration::from_secs(60 * 2); // 15 minutes.
        let mut workers = Vec::new();
        let mut reqs: Vec<Vec<u8>> = Vec::new();
        for i in 0..3 {
            let img_name = if i % 3 == 0 {
                "lion.jpeg"
            } else if i % 3 == 1 {
                "elephant.jpeg"
            } else {
                "panda.jpeg"
            };
            let img = ImageReader::open(img_name).unwrap().decode().unwrap();
            let req = InferReq {
                width: img.width(),
                height: img.height(),
                img: general_purpose::STANDARD_NO_PAD.encode(img.into_bytes()),
            };
            let req = serde_json::to_vec(&req).unwrap();
            reqs.push(req);
        }
        for n in 0..num_workers {
            let fc = fc.clone();
            let test_duration = test_duration.clone();
            let reqs = reqs.clone();
            workers.push(tokio::spawn(async move {
                let mut results: Vec<(u64, f64, String)> = Vec::new();
                let start_time = std::time::Instant::now();
                let mut i = 0;
                loop {
                    i += 1;
                    // Pick an image at random.
                    // TODO: Find a better way to select images.
                    let curr_time = std::time::Instant::now();
                    let since = curr_time.duration_since(start_time);
                    if since > test_duration {
                        break;
                    }
                    let since = since.as_millis() as u64;
                    let req = &reqs[i % 3];
                    let resp = fc.invoke(req).await;
                    if resp.is_err() {
                        println!("Err: {resp:?}");
                        continue;
                    }
                    let resp = resp.unwrap();
                    let resp: Vec<(std::time::Duration, (Value, bool))> =
                        serde_json::from_value(resp).unwrap();
                    let resp: Vec<_> = resp
                        .into_iter()
                        .map(|(x, y)| (x.as_secs_f64(), y))
                        .collect();
                    let end_time = std::time::Instant::now();
                    let active_time_ms = end_time.duration_since(start_time).as_secs_f64() * 1000.0;

                    if n < 2 {
                        // Avoid too many prints.
                        println!("Worker {n} Resp: {resp:?}");
                        println!("Worker {n} Since: {since:?}");
                    }
                    for (duration, (_, is_direct)) in resp {
                        let mode = if is_direct {
                            "ECS".to_string()
                        } else {
                            "Lambda".to_string()
                        };
                        results.push((since, duration, mode));
                    }
                    let mut wait_time_ms = active_time_ms / activity - active_time_ms;
                    if wait_time_ms > 10.0 * 1000.0 {
                        wait_time_ms = 10.0 * 1000.0; // Prevent excessive waiting.
                    }
                    if wait_time_ms > 1.0 {
                        let wait_time =
                            std::time::Duration::from_millis(wait_time_ms.ceil() as u64);
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
    async fn simple_bench_cloud() {
        let fc = Arc::new(FunctionalClient::new("benchinfer").await);
        let img = ImageReader::open("lion.jpeg").unwrap().decode().unwrap();
        let req = InferReq {
            width: img.width(),
            height: img.height(),
            img: general_purpose::STANDARD_NO_PAD.encode(img.into_bytes()),
        };

        let req = serde_json::to_vec(&req).unwrap();
        let resp = fc.invoke(&req).await.unwrap();
        let resp: Vec<(std::time::Duration, Value)> = serde_json::from_value(resp).unwrap();
        println!("Response: {resp:?}");
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 16)]
    async fn full_bench_cloud() {
        let fc = Arc::new(FunctionalClient::new("benchinfer").await);
        // run_bench(fc.clone(), RequestRate::Low, "pre").await;
        run_bench(fc.clone(), RequestRate::Medium, "pre").await;
        run_bench(fc.clone(), RequestRate::High(10), "pre").await;
        // run_bench(fc.clone(), RequestRate::Low, "post").await;
    }
}
