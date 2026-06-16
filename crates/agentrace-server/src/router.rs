// ======================================================================
// `ROUTER`
//
// 1. Axum router builder — assembles all API and static routes.
// 2. Embeds the frontend SPA via rust-embed for single-binary distribution.
// 3. Modification history:
//    - 16 June 2025: Initial skeleton
//
//     Author: Zi Liang <zi1415926.liang@connect.polyu.hk>
//     Copyright © 2025, Zi Liang, all rights reserved.
//     Created: 16 June 2025
// ======================================================================

use axum::{Router, body::Body, http::{StatusCode, header}, response::Response, routing::get, Json};
use rust_embed::Embed;
use serde_json::{json, Value};
use std::net::SocketAddr;

/// Embedded frontend build artifacts from `frontend/dist/`.
#[derive(Embed)]
#[folder = "../../frontend/dist/"]
struct FrontendAssets;

pub fn build_router() -> Router {
    Router::new()
        .route("/api/v1/health", get(health_check))
        .route("/api/v1/stats", get(stats_stub))
        .nest("/api/v1/", crate::api::routes())
        // Serve embedded frontend as fallback for SPA routing
        .fallback(get(serve_frontend))
}

async fn health_check() -> Json<Value> {
    Json(json!({"status": "ok", "service": "agentrace"}))
}

async fn stats_stub() -> Json<Value> {
    Json(json!({
        "utterances": 0,
        "conversations": 0,
        "agents": []
    }))
}

/// Serve the embedded frontend SPA.
async fn serve_frontend(uri: axum::http::Uri) -> Result<Response, StatusCode> {
    let path = uri.path().trim_start_matches('/');
    let path = if path.is_empty() { "index.html" } else { path };

    match <FrontendAssets as Embed>::get(path) {
        Some(content) => {
            let mime = mime_guess::from_path(path).first_or_octet_stream();
            Ok(Response::builder()
                .status(StatusCode::OK)
                .header(header::CONTENT_TYPE, mime.as_ref())
                .body(Body::from(content.data.into_owned()))
                .unwrap())
        }
        None => {
            // SPA fallback: serve index.html for client-side routing
            match <FrontendAssets as Embed>::get("index.html") {
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

pub async fn serve(addr: SocketAddr) -> anyhow::Result<()> {
    let app = build_router();
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

    fn app() -> Router {
        build_router()
    }

    #[tokio::test]
    async fn health_check_returns_ok() {
        let response = app()
            .oneshot(
                Request::builder()
                    .uri("/api/v1/health")
                    .body(axum::body::Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::OK);

        let body = axum::body::to_bytes(response.into_body(), 1024)
            .await
            .unwrap();
        let json: Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(json["status"], "ok");
        assert_eq!(json["service"], "agentrace");
    }

    #[tokio::test]
    async fn stats_endpoint_returns_json() {
        let response = app()
            .oneshot(
                Request::builder()
                    .uri("/api/v1/stats")
                    .body(axum::body::Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::OK);

        let body = axum::body::to_bytes(response.into_body(), 1024)
            .await
            .unwrap();
        let json: Value = serde_json::from_slice(&body).unwrap();
        assert!(json["utterances"].is_number());
        assert!(json["conversations"].is_number());
    }

    #[tokio::test]
    async fn frontend_serves_index_html() {
        let response = app()
            .oneshot(
                Request::builder()
                    .uri("/")
                    .body(axum::body::Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::OK);

        let body = axum::body::to_bytes(response.into_body(), 10_000)
            .await
            .unwrap();
        let html = String::from_utf8_lossy(&body);
        assert!(html.contains("<!doctype html") || html.contains("<!DOCTYPE html"));
    }

    #[tokio::test]
    async fn frontend_spa_fallback() {
        let response = app()
            .oneshot(
                Request::builder()
                    .uri("/some-client-route")
                    .body(axum::body::Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();

        // SPA fallback: should serve index.html, not 404
        assert_eq!(response.status(), StatusCode::OK);

        let body = axum::body::to_bytes(response.into_body(), 10_000)
            .await
            .unwrap();
        let html = String::from_utf8_lossy(&body);
        assert!(html.contains("<!doctype html") || html.contains("<!DOCTYPE html"));
    }

    #[tokio::test]
    async fn api_routes_are_registered() {
        let response = app()
            .oneshot(
                Request::builder()
                    .uri("/api/v1/health")
                    .method("GET")
                    .body(axum::body::Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn invalid_api_route_returns_fallback_not_404() {
        // Non-existent API route — since there's a blanket fallback to index.html,
        // we get the SPA, not a 404
        let response = app()
            .oneshot(
                Request::builder()
                    .uri("/api/v1/nonexistent")
                    .body(axum::body::Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();

        // Current behaviour: fallback serves index.html
        // KEY REVIEW POINT: Should unknown API routes return 404 instead of SPA fallback?
        assert_eq!(response.status(), StatusCode::OK);
    }
}
