#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // TODO: Replace with "deployment.toml" when done testing.
    let main_deployment = include_str!("deployment.toml");
    let bench_deployment = include_str!("bench_depl.toml");
    let deployments: Vec<String> = vec![main_deployment.into(), bench_deployment.into()];
    obelisk_deployment::build_user_deployment(
        "inference",
        "public.ecr.aws/c6l0e8f4/obk-img-system",
        &deployments,
    )
    .await;
    Ok(())
}
