// ======================================================================
// `ROUTER`
//
// 1. Axum router builder — assembles all API and static routes.
// 2. Embeds the frontend SPA via rust-embed for single-binary distribution.
// 3. Modification history:
//    - 16 June 2025: Initial skeleton
//    - 16 June 2025: Phase 3 — AppState with real Store
//
//     Author: Zi Liang <zi1415926.liang@connect.polyu.hk>
//     Copyright © 2025, Zi Liang, all rights reserved.
//     Created: 16 June 2025
// ======================================================================

use axum::{
    Router,
    body::Body,
    http::{StatusCode, header},
    response::Response,
    routing::get,
    Json,
};
use rust_embed::Embed;
use serde_json::{json, Value};
use std::net::SocketAddr;
use std::sync::Arc;

use crate::api::AppState;

/// Embedded frontend build artifacts from `frontend/dist/`.
#[derive(Embed)]
#[folder = "../../frontend/dist/"]
struct FrontendAssets;

pub fn build_router(state: Arc<AppState>) -> Router {
    Router::new()
        .route("/api/v1/health", get(health_check))
        .route("/api/v1/stats", get(move |s| crate::api::get_stats(s)))
        .route("/api/v1/analysis", get(move |s| crate::api::get_analysis(s)))
        .route("/api/v1/utterances", get(move |s| crate::api::get_utterances(s)))
        .route("/api/v1/graph", get(move |s| crate::api::get_graph(s)))
        .nest("/api/v1/", crate::api::routes())
        .with_state(state)
        .fallback(get(serve_frontend))
}

async fn health_check() -> Json<Value> {
    Json(json!({"status": "ok", "service": "agentrace"}))
}

async fn serve_frontend(uri: axum::http::Uri) -> Result<Response, StatusCode> {
    let path = uri.path().trim_start_matches('/');
    let path = if path.is_empty() { "index.html" } else { path };

    match FrontendAssets::get(path) {
        Some(content) => {
            let mime = mime_guess::from_path(path).first_or_octet_stream();
            Ok(Response::builder()
                .status(StatusCode::OK)
                .header(header::CONTENT_TYPE, mime.as_ref())
                .body(Body::from(content.data.into_owned()))
                .unwrap())
        }
        None => {
            match FrontendAssets::get("index.html") {
                Some(content) => Ok(Response::builder()
                    .status(StatusCode::OK)
                    .header(header::CONTENT_TYPE, "text/html")
                    .body(Body::from(content.data.into_owned()))
                    .unwrap()),
                None => Err(StatusCode::NOT_FOUND),
            }
        }
    }
}

pub async fn serve(addr: SocketAddr, store: agentrace_storage::Store) -> anyhow::Result<()> {
    let state = Arc::new(AppState { store });
    let app = build_router(state);
    tracing::info!("Agentrace dashboard starting on http://{}", addr);

    let listener = tokio::net::TcpListener::bind(addr).await?;
    axum::serve(listener, app).await?;

    Ok(())
}

// ======================================================================
// Tests
// ======================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use axum::http::{Request, StatusCode};
    use tower::ServiceExt;

    fn test_app() -> Router {
        let store = agentrace_storage::Store::open(":memory:").unwrap();
        let state = Arc::new(AppState { store });
        build_router(state)
    }

    #[tokio::test]
    async fn health_check_returns_ok() {
        let response = test_app()
            .oneshot(
                Request::builder()
                    .uri("/api/v1/health")
                    .body(axum::body::Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::OK);
        let body = axum::body::to_bytes(response.into_body(), 1024).await.unwrap();
        let json: Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(json["status"], "ok");
    }

    #[tokio::test]
    async fn stats_endpoint_returns_json() {
        let response = test_app()
            .oneshot(
                Request::builder()
                    .uri("/api/v1/stats")
                    .body(axum::body::Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::OK);
        let body = axum::body::to_bytes(response.into_body(), 4096).await.unwrap();
        let json: Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(json["utterances"], 0);
        assert_eq!(json["conversations"], 0);
    }

    #[tokio::test]
    async fn frontend_serves_index_html() {
        let response = test_app()
            .oneshot(
                Request::builder()
                    .uri("/")
                    .body(axum::body::Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::OK);
    }
}
