use base64::{engine::general_purpose, Engine as _};
use obelisk::FunctionInstance;
use pyo3::prelude::*;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::sync::{Arc, Mutex};
use tokio::runtime::Handle;
use tokio::sync::{mpsc, oneshot};

mod bench_fn;
mod infer_test;
pub use bench_fn::BenchFn;

#[derive(Debug)]
struct InferCmd {
    img: Vec<u8>,
    width: u32,
    height: u32,
    resp_tx: oneshot::Sender<String>,
}

#[derive(Serialize, Deserialize, Clone)]
struct InferReq {
    width: u32,
    height: u32,
    img: String, // Base64 encoded image.
}

#[derive(Serialize, Deserialize, Clone, Debug)]
struct InferResp {
    label: String,
}

#[derive(Clone)]
pub struct InferFn {
    tx: mpsc::Sender<InferCmd>,
    initialized: Arc<Mutex<bool>>,
}

#[async_trait::async_trait]
impl FunctionInstance for InferFn {
    async fn invoke(&self, arg: Value) -> Value {
        println!("Received Inference Invoked");
        let initialized: bool = {
            let initialied = self.initialized.lock().unwrap();
            *initialied
        };
        if !initialized {
            tokio::time::sleep(std::time::Duration::from_secs(1)).await;
            let initialized: bool = {
                let initialized = self.initialized.lock().unwrap();
                *initialized
            };
            if !initialized {
                panic!("Inference function still reading hardware specs!");
            }
        }
        let req: InferReq = serde_json::from_value(arg).unwrap();
        println!(
            "Decoded Inference Request: {}, {}, {}",
            req.width,
            req.height,
            req.img.len()
        );
        let resp = self.handle_request(req).await;
        let resp = serde_json::to_value(&resp).unwrap();
        resp
    }
}

impl InferFn {
    /// Create new inferer.
    pub async fn new() -> Self {
        let (tx, rx) = mpsc::channel(10);
        let initialized = Arc::new(Mutex::new(false));
        let handle = Handle::current();
        let initialized1 = initialized.clone();
        tokio::task::spawn_blocking(move || {
            Self::start_inferer(rx, initialized1.clone(), handle);
        });

        InferFn { tx, initialized }
    }

    /// Handle request.
    async fn handle_request(&self, req: InferReq) -> InferResp {
        let InferReq { width, height, img } = req;
        let (resp_tx, resp_rx) = oneshot::channel();
        // let start_time = std::time::Instant::now();
        let cmd = InferCmd {
            width,
            height,
            img: general_purpose::STANDARD_NO_PAD.decode(img).unwrap(),
            resp_tx,
        };
        let tx = self.tx.clone();
        tx.send(cmd).await.unwrap();
        // let end_time = std::time::Instant::now();
        // let duration = end_time.duration_since(start_time);
        // println!("Passing requst. Duration: {duration:?}");
        // let start_time = std::time::Instant::now();
        let label: String = resp_rx.await.unwrap();
        let resp = InferResp { label };
        // let end_time = std::time::Instant::now();
        // let duration = end_time.duration_since(start_time);
        // println!("Getting Response. Duration: {duration:?}");
        resp
    }

    /// Start inferer thread.
    fn start_inferer(
        mut rx: mpsc::Receiver<InferCmd>,
        initialized: Arc<Mutex<bool>>,
        handle: Handle,
    ) {
        println!("Starting inferer thread!");
        let py_code = include_str!("main.py");
        let _res: PyResult<()> = Python::with_gil(|py| {
            let fun: Py<PyAny> = PyModule::from_code(py, py_code, "", "")
                .unwrap()
                .getattr("main")
                .unwrap()
                .into();
            {
                let mut initialized = initialized.lock().unwrap();
                *initialized = true;
            }
            loop {
                // let start_time = std::time::Instant::now();
                let mut cmds = Vec::new();
                println!("Inferer loop running!");
                // Read first cmd.
                let cmd = handle.block_on(rx.recv());
                let cmd = match cmd {
                    None => return Ok(()),
                    Some(cmd) => cmd,
                };
                cmds.push(cmd);
                // Read all pending commands into a batch.
                loop {
                    let cmd = rx.try_recv();
                    if let Ok(cmd) = cmd {
                        cmds.push(cmd);
                    } else {
                        break;
                    }
                }
                let inputs: Vec<_> = cmds
                    .iter()
                    .map(|cmd| (cmd.img.as_slice(), cmd.width, cmd.height))
                    .collect();
                // let end_time = std::time::Instant::now();
                // let duration = end_time.duration_since(start_time);
                // println!("Read Inputs. Duration: {duration:?}");
                // let start_time = std::time::Instant::now();
                let res: Py<PyAny> = fun.call1(py, (inputs,)).unwrap();
                let results: Vec<String> = res.extract(py).unwrap();
                println!("Inferer loop found: {res}!");
                // let end_time = std::time::Instant::now();
                // let duration = end_time.duration_since(start_time);
                // println!("Run code. Duration: {duration:?}");
                // let start_time = std::time::Instant::now();
                for (cmd, res) in cmds.into_iter().zip(results) {
                    cmd.resp_tx.send(res).unwrap();
                }
                // let end_time = std::time::Instant::now();
                // let duration = end_time.duration_since(start_time);
                // println!("Respond. Duration: {duration:?}");
            }
        });
    }
}
