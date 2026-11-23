//! `fxraw serve` — a local web UI for browsing recipes and converting a RAF
//! to any recipe on the camera, with before/after preview.
//!
//! Everything runs on 127.0.0.1. Uploaded RAFs and rendered JPEGs live in a
//! temporary cache directory; "Save" copies a render next to the served folder.

use crate::{analyse, fuji, matcher, recipes, ui};
use serde_json::{json, Value};
use std::io::Read;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Instant;
use tiny_http::{Header, Method, Request, Response, Server};

static INDEX_HTML: &str = include_str!("web/index.html");

/// Uploads larger than this are rejected (X100VI RAFs are ~30–60 MB).
const MAX_UPLOAD_BYTES: u64 = 512 * 1024 * 1024;

/// JSON bodies (edited recipes) larger than this are rejected.
const MAX_JSON_BYTES: u64 = 64 * 1024;

struct Image {
    name: String,
    path: PathBuf,
    uploaded: bool,
}

struct State {
    /// Folder whose RAFs are listed, and where saved JPEGs are written.
    root: PathBuf,
    /// Temporary working directory (uploads, previews, renders).
    cache: PathBuf,
    /// Registered images; the id is the index.
    images: Mutex<Vec<Image>>,
    /// Serialises access to the camera (one conversion at a time).
    camera: Mutex<()>,
    /// Description of the conversion currently running, if any.
    busy: Mutex<Option<String>>,
    /// Reference-matching jobs; the id is the index.
    matches: Mutex<Vec<Arc<MatchJob>>>,
    /// Digital tele-converter crop per RAF (see `tele_crop`), read once.
    crops: Mutex<std::collections::HashMap<PathBuf, Option<[f32; 4]>>>,
}

/// A "match this reference" search running (or finished) in the background.
struct MatchJob {
    id: usize,
    image: usize,
    /// File name of the reference picture (e.g. the film scan).
    reference: String,
    /// Where the aligned reference and the step previews are written.
    dir: PathBuf,
    stop: AtomicBool,
    status: Mutex<MatchStatus>,
}

struct MatchStatus {
    /// "aligning" | "running" | "done" | "stopped" | "error"
    state: &'static str,
    phase: String,
    error: Option<String>,
    alignment: Option<Value>,
    cells: usize,
    dropped: usize,
    monochrome: bool,
    steps: Vec<Value>,
    /// (render number, loss) of the best render so far.
    best: Option<(usize, f32)>,
    best_recipe: Option<Value>,
    best_params: Option<matcher::Params>,
    best_label: String,
    max_renders: usize,
    started: Instant,
    finished: Option<f64>,
    /// Wall time of each camera render, for the ETA.
    render_secs: Vec<f64>,
    budget_exhausted: bool,
    /// Set while the camera stopped answering and we wait for it to return.
    waiting_camera: Option<String>,
}

type Resp = Response<std::io::Cursor<Vec<u8>>>;

pub fn serve(root: &Path, port: u16, open_browser: bool) {
    let root = root.canonicalize().unwrap_or_else(|e| {
        eprintln!("Error: cannot open folder '{}': {e}", root.display());
        std::process::exit(1);
    });
    // One cache per port, wiped on start so renders from old runs don't pile up.
    let mut cache = std::env::temp_dir().join(format!("fxraw-web-{port}"));
    let _ = std::fs::remove_dir_all(&cache);
    if cache.exists() {
        // Left behind by another user (e.g. a previous `sudo fxraw serve`).
        cache = std::env::temp_dir().join(format!("fxraw-web-{port}-{}", std::process::id()));
    }
    for sub in ["uploads", "previews", "renders", "matches"] {
        if let Err(e) = std::fs::create_dir_all(cache.join(sub)) {
            eprintln!("Error: cannot create cache dir '{}': {e}", cache.display());
            std::process::exit(1);
        }
    }

    let addr = format!("127.0.0.1:{port}");
    let server = Server::http(&addr).unwrap_or_else(|e| {
        eprintln!("Error: cannot listen on {addr}: {e}");
        eprintln!("Try another port with --port.");
        std::process::exit(1);
    });
    let url = format!("http://{addr}/");
    println!("fxraw web UI running at {url}");
    println!("  folder: {}", root.display());
    println!("  cache:  {}", cache.display());
    println!("Press Ctrl+C to stop.\n");
    if open_browser {
        open_url(&url);
    }

    let state = Arc::new(State {
        root,
        cache,
        images: Mutex::new(Vec::new()),
        camera: Mutex::new(()),
        busy: Mutex::new(None),
        matches: Mutex::new(Vec::new()),
        crops: Mutex::new(std::collections::HashMap::new()),
    });

    for req in server.incoming_requests() {
        let state = Arc::clone(&state);
        std::thread::spawn(move || handle(&state, req));
    }
}

/// Open the URL in the default browser — as the invoking user when under sudo.
fn open_url(url: &str) {
    let mut cmd = match std::env::var("SUDO_USER") {
        Ok(user) if cfg!(target_os = "macos") => {
            let mut c = std::process::Command::new("sudo");
            c.args(["-u", &user, "open", url]);
            c
        }
        _ if cfg!(target_os = "macos") => {
            let mut c = std::process::Command::new("open");
            c.arg(url);
            c
        }
        _ => {
            let mut c = std::process::Command::new("xdg-open");
            c.arg(url);
            c
        }
    };
    let _ = cmd.spawn();
}

// ---------------------------------------------------------------------------
// Routing
// ---------------------------------------------------------------------------

fn handle(state: &Arc<State>, mut req: Request) {
    let method = req.method().clone();
    let full_url = req.url().to_string();
    let (path, query) = full_url.split_once('?').unwrap_or((&full_url, ""));
    let segs: Vec<&str> = path.trim_matches('/').split('/').filter(|s| !s.is_empty()).collect();

    let resp = match (&method, segs.as_slice()) {
        (Method::Get, []) => {
            Response::from_data(INDEX_HTML.as_bytes().to_vec()).with_header(content_type("text/html; charset=utf-8"))
        }
        (Method::Get, ["api", "status"]) => json_ok(status(state)),
        (Method::Get, ["api", "recipes"]) => json_ok(recipes_json()),
        (Method::Post, ["api", "my-recipes"]) => save_my_recipe(state, &mut req),
        (Method::Delete, ["api", "my-recipes", slug]) => delete_my_recipe(state, slug),
        (Method::Get, ["api", "files"]) => json_ok(list_folder(state)),
        (Method::Post, ["api", "upload"]) => upload(state, &mut req),
        (Method::Get, ["api", "images", id]) => with_image(state, id, |id| image_json(state, id)),
        (Method::Get, ["api", "images", id, "preview"]) => {
            with_image_resp(state, id, |id| preview(state, id, false))
        }
        (Method::Get, ["api", "images", id, "thumb"]) => {
            with_image_resp(state, id, |id| preview(state, id, true))
        }
        (Method::Get, ["api", "images", id, "analysis"]) => {
            with_image_resp(state, id, |id| analysis(state, id))
        }
        (Method::Post, ["api", "images", id, "convert", slug]) => {
            with_image_resp(state, id, |id| convert(state, id, slug))
        }
        (Method::Post, ["api", "images", id, "convert-custom"]) => {
            with_image_resp(state, id, |id| convert_custom(state, id, &mut req))
        }
        (Method::Get, ["api", "images", id, "renders", file]) => {
            let download = query.split('&').any(|kv| kv == "download=1");
            with_image_resp(state, id, |id| render_file(state, id, file, download))
        }
        (Method::Post, ["api", "images", id, "renders", slug, "save"]) => {
            with_image_resp(state, id, |id| save_render(state, id, slug))
        }
        (Method::Post, ["api", "images", id, "match"]) => {
            with_image_resp(state, id, |id| start_match(state, id, &mut req, query))
        }
        (Method::Get, ["api", "images", id, "match"]) => with_image(state, id, |id| {
            let jobs = state.matches.lock().unwrap();
            jobs.iter().rev().find(|j| j.image == id).map(|j| match_json(j)).unwrap_or(Value::Null)
        }),
        (Method::Get, ["api", "match", job]) => with_match(state, job, |j| json_ok(match_json(&j))),
        (Method::Get, ["api", "match", job, file]) => {
            let download = query.split('&').any(|kv| kv == "download=1");
            with_match(state, job, |j| match_file(state, &j, file, download))
        }
        (Method::Post, ["api", "match", job, "stop"]) => with_match(state, job, |j| {
            j.stop.store(true, Ordering::Relaxed);
            json_ok(json!({ "ok": true }))
        }),
        (Method::Post, ["api", "match", job, "save"]) => with_match(state, job, |j| save_match(state, &j)),
        _ => json_err(404, "not found"),
    };
    let _ = req.respond(resp);
}

fn with_image(state: &State, id: &str, f: impl FnOnce(usize) -> Value) -> Resp {
    with_image_resp(state, id, |id| json_ok(f(id)))
}

fn with_image_resp(state: &State, id: &str, f: impl FnOnce(usize) -> Resp) -> Resp {
    match id.parse::<usize>() {
        Ok(id) if id < state.images.lock().unwrap().len() => f(id),
        _ => json_err(404, "unknown image"),
    }
}

// ---------------------------------------------------------------------------
// Handlers
// ---------------------------------------------------------------------------

fn status(state: &State) -> Value {
    json!({
        "camera": camera_connected(),
        "stub": std::env::var("FXRAW_STUB_CAMERA").is_ok(),
        "busy": *state.busy.lock().unwrap(),
        "exiftool": which("exiftool"),
        "folder": state.root.display().to_string(),
    })
}

/// List RAFs in the served folder and its `_RAF` subfolder (project layout).
fn list_folder(state: &State) -> Value {
    let mut found: Vec<PathBuf> = Vec::new();
    for dir in [state.root.clone(), state.root.join("_RAF")] {
        if let Ok(entries) = std::fs::read_dir(&dir) {
            for e in entries.flatten() {
                let p = e.path();
                let is_raf = p
                    .extension()
                    .is_some_and(|x| x.eq_ignore_ascii_case("raf"));
                if is_raf && p.is_file() {
                    found.push(p);
                }
            }
        }
    }
    found.sort();
    for p in found {
        let name = p.file_name().unwrap_or_default().to_string_lossy().into_owned();
        register(state, name, p, false);
    }
    // Everything known, uploads included, so a page reload finds them again.
    let n = state.images.lock().unwrap().len();
    json!((0..n).map(|id| image_json(state, id)).collect::<Vec<_>>())
}

fn register(state: &State, name: String, path: PathBuf, uploaded: bool) -> usize {
    let mut images = state.images.lock().unwrap();
    if let Some(i) = images.iter().position(|im| im.path == path) {
        return i;
    }
    images.push(Image { name, path, uploaded });
    images.len() - 1
}

fn image_json(state: &State, id: usize) -> Value {
    let (name, rel, uploaded) = {
        let images = state.images.lock().unwrap();
        let im = &images[id];
        let rel = im
            .path
            .strip_prefix(&state.root)
            .map(|p| p.display().to_string())
            .unwrap_or_else(|_| im.name.clone());
        (im.name.clone(), rel, im.uploaded)
    };
    let mut renders: Vec<String> = std::fs::read_dir(state.cache.join("renders").join(id.to_string()))
        .map(|rd| {
            rd.flatten()
                .filter_map(|e| {
                    let n = e.file_name().to_string_lossy().into_owned();
                    n.strip_suffix(".jpg").map(str::to_string)
                })
                .collect()
        })
        .unwrap_or_default();
    renders.sort();
    // Edited versions: what they were made from, so a reload can label them.
    let custom: serde_json::Map<String, Value> = renders
        .iter()
        .filter(|k| is_custom_key(k))
        .filter_map(|k| {
            let meta = std::fs::read(render_path(state, id, &format!("{k}.json"))).ok()?;
            Some((k.clone(), serde_json::from_slice(&meta).ok()?))
        })
        .collect();
    json!({ "id": id, "name": name, "path": rel, "uploaded": uploaded, "renders": renders, "custom": custom })
}

fn upload(state: &State, req: &mut Request) -> Resp {
    let raw_name = req
        .headers()
        .iter()
        .find(|h| h.field.equiv("X-Filename"))
        .map(|h| percent_decode(h.value.as_str()))
        .unwrap_or_else(|| "upload.RAF".into());
    let name: String = Path::new(&raw_name)
        .file_name()
        .map(|n| n.to_string_lossy().into_owned())
        .unwrap_or_else(|| "upload.RAF".into())
        .chars()
        .map(|c| if c.is_ascii_alphanumeric() || "._-".contains(c) { c } else { '_' })
        .collect();

    let mut data = Vec::new();
    if let Err(e) = req.as_reader().take(MAX_UPLOAD_BYTES + 1).read_to_end(&mut data) {
        return json_err(400, &format!("upload failed: {e}"));
    }
    if data.len() as u64 > MAX_UPLOAD_BYTES {
        return json_err(413, "file too large");
    }
    if data.len() < 16 || &data[..15] != b"FUJIFILMCCD-RAW" {
        return json_err(400, "not a Fujifilm RAF file");
    }

    let n = state.images.lock().unwrap().len();
    let path = state.cache.join("uploads").join(format!("{n}-{name}"));
    if let Err(e) = std::fs::write(&path, &data) {
        return json_err(500, &format!("could not store upload: {e}"));
    }
    let id = register(state, name.clone(), path, true);
    println!("uploaded {name} ({:.1} MB)", data.len() as f64 / 1_048_576.0);
    json_ok(image_json(state, id))
}

/// The camera's own JPEG embedded in the RAF (as shot). `thumb` returns the
/// small EXIF thumbnail inside it when present, for fast lists.
fn preview(state: &State, id: usize, thumb: bool) -> Resp {
    let kind = if thumb { "thumb" } else { "full" };
    let cached = state.cache.join("previews").join(format!("{id}-{kind}.jpg"));
    if !cached.exists() {
        let path = state.images.lock().unwrap()[id].path.clone();
        let raf = match std::fs::read(&path) {
            Ok(d) => d,
            Err(e) => return json_err(500, &format!("cannot read RAF: {e}")),
        };
        let Some(full) = fuji::embedded_jpeg(&raf) else {
            return json_err(422, "no embedded preview in this RAF");
        };
        // The preview always covers the whole sensor; with the digital
        // tele-converter the camera crops when it develops, so show the
        // same framing. Thumbnails are made from the preview because the
        // EXIF thumbnail is letterboxed 4:3 and has no orientation tag.
        let crop = tele_crop(state, &path);
        let bytes: Vec<u8> = if thumb {
            match upright_preview(full, crop) {
                Some(img) => matcher::encode_jpeg(&matcher::thumbnail(&img, 320), 82),
                None => exif_thumbnail(full).unwrap_or(full).to_vec(),
            }
        } else if crop.is_some() {
            upright_preview(full, crop).map(|img| matcher::encode_jpeg(&img, 90)).unwrap_or_else(|| full.to_vec())
        } else {
            full.to_vec() // browsers apply its EXIF orientation themselves
        };
        if let Err(e) = std::fs::write(&cached, bytes) {
            return json_err(500, &format!("cache write failed: {e}"));
        }
    }
    file_response(&cached, "image/jpeg", None)
}

fn analysis(state: &State, id: usize) -> Resp {
    let path = state.images.lock().unwrap()[id].path.clone();
    let tele = tele_crop(state, &path).map(|c| (10.0 / c[2] as f64).round() / 10.0);
    match analyse::analyse_file(&path, 5) {
        Ok(a) => json_ok(json!({
            "tele": tele,
            "settings": a.settings.iter().map(|(k, v)| json!([k, v])).collect::<Vec<_>>(),
            "matches": a.matches.iter().map(|m| json!({
                "slug": m.recipe.slug,
                "name": m.recipe.name,
                "percent": (m.percent * 10.0).round() / 10.0,
            })).collect::<Vec<_>>(),
        })),
        Err(e) => json_err(503, &e),
    }
}

fn convert(state: &State, id: usize, slug: &str) -> Resp {
    let Some(recipe) = recipes::all().iter().find(|r| r.slug == slug).cloned() else {
        return json_err(404, "unknown recipe");
    };
    render(state, id, slug, &recipe.name, &recipe.to_settings())
}

/// Render an edited recipe (sent as JSON in the body). The result is cached
/// under a key derived from the settings, so the same edit isn't rendered twice.
fn convert_custom(state: &State, id: usize, req: &mut Request) -> Resp {
    let recipe = match read_json::<recipes::Recipe>(req) {
        Ok(r) => r,
        Err(e) => return json_err(400, &e),
    };
    if let Err(e) = recipe.validate() {
        return json_err(400, &e);
    }
    let key = custom_key(&recipe);
    let label = format!("{} (edited)", recipe.name);
    let meta = render_path(state, id, &format!("{key}.json"));
    if !meta.exists() {
        let doc = json!({ "label": label, "recipe": recipe });
        if let Err(e) = std::fs::create_dir_all(meta.parent().unwrap()).and_then(|_| std::fs::write(&meta, doc.to_string())) {
            return json_err(500, &format!("could not store render: {e}"));
        }
    }
    render(state, id, &key, &label, &recipe.to_settings())
}

/// "custom-<16 hex>": a stable hash of everything that affects the render.
fn custom_key(recipe: &recipes::Recipe) -> String {
    use std::hash::{Hash, Hasher};
    let mut r = recipe.clone();
    // Name, source and shooting hints don't change the picture.
    r.name.clear();
    r.slug.clear();
    r.source = None;
    r.iso = None;
    r.exposure_comp = None;
    let mut h = std::hash::DefaultHasher::new();
    serde_json::to_string(&r).unwrap_or_default().hash(&mut h);
    format!("custom-{:016x}", h.finish())
}

fn is_custom_key(key: &str) -> bool {
    key.strip_prefix("custom-").is_some_and(|h| h.len() == 16 && h.bytes().all(|b| b.is_ascii_hexdigit()))
}

/// A render the server knows: a recipe slug or an edited version's key.
fn is_render_key(key: &str) -> bool {
    is_known_slug(key) || is_custom_key(key)
}

fn render_path(state: &State, id: usize, file: &str) -> PathBuf {
    state.cache.join("renders").join(id.to_string()).join(file)
}

/// Convert image `id` with `settings` on the camera into `renders/<id>/<key>.jpg`.
fn render(state: &State, id: usize, key: &str, label: &str, settings: &crate::profile::RecipeSettings) -> Resp {
    let out = render_path(state, id, &format!("{key}.jpg"));
    if out.exists() {
        return json_ok(json!({ "key": key, "cached": true, "warnings": [] }));
    }
    let (name, path) = {
        let images = state.images.lock().unwrap();
        (images[id].name.clone(), images[id].path.clone())
    };

    let _guard = state.camera.lock().unwrap();
    // A queued duplicate request may have produced it while we waited.
    if out.exists() {
        return json_ok(json!({ "key": key, "cached": true, "warnings": [] }));
    }
    *state.busy.lock().unwrap() = Some(format!("{name} → {label}"));
    println!("converting {name} → {label} …");
    let started = Instant::now();

    let result = fuji::try_open_camera().and_then(|mut camera| {
        fuji::convert_raf_to_jpeg(&mut *camera, &path.to_string_lossy(), settings, &ui::ConvertProgress::quiet())
    });
    *state.busy.lock().unwrap() = None;
    let secs = started.elapsed().as_secs_f64();

    match result {
        Ok((jpeg, warnings)) => {
            if let Err(e) = std::fs::create_dir_all(out.parent().unwrap()).and_then(|_| std::fs::write(&out, &jpeg)) {
                return json_err(500, &format!("could not store render: {e}"));
            }
            println!("  done in {secs:.1}s ({:.1} MB)", jpeg.len() as f64 / 1_048_576.0);
            for w in &warnings {
                println!("  {w}");
            }
            json_ok(json!({ "key": key, "cached": false, "seconds": secs, "warnings": warnings }))
        }
        Err(e) => {
            println!("  failed: {e}");
            json_err(502, &e)
        }
    }
}

/// File name stem for a downloaded or saved render: `<raf stem>-<recipe>`.
fn render_file_stem(state: &State, id: usize, key: &str) -> String {
    let what = if is_custom_key(key) {
        std::fs::read(render_path(state, id, &format!("{key}.json")))
            .ok()
            .and_then(|m| serde_json::from_slice::<Value>(&m).ok())
            .and_then(|m| m["label"].as_str().map(recipes::slugify))
            .unwrap_or_else(|| key.to_string())
    } else {
        key.to_string()
    };
    format!("{}-{what}", image_stem(state, id))
}

/// All recipes, the user's own flagged with `"mine": true`.
fn recipes_json() -> Value {
    let all = recipes::all();
    Value::Array(
        all.iter()
            .map(|r| {
                let mut v = json!(r);
                if r.mine {
                    v["mine"] = json!(true);
                }
                v
            })
            .collect(),
    )
}

/// Save the posted recipe as one of the user's own.
/// Body: `{ recipe, overwrite?, image?, from? }`. When `image` and `from` (a
/// render key of that image) are given, that render becomes the new recipe's
/// render so it doesn't have to be converted again.
fn save_my_recipe(state: &State, req: &mut Request) -> Resp {
    let body = match read_json::<Value>(req) {
        Ok(b) => b,
        Err(e) => return json_err(400, &e),
    };
    let recipe: recipes::Recipe = match serde_json::from_value(body["recipe"].clone()) {
        Ok(r) => r,
        Err(e) => return json_err(400, &format!("invalid recipe: {e}")),
    };
    let overwrite = body["overwrite"].as_bool().unwrap_or(false);
    let saved = match recipes::save_mine(recipe, overwrite) {
        Ok(r) => r,
        Err(e @ recipes::SaveError::Exists(_)) => {
            return Response::from_data(json!({ "error": e.to_string(), "exists": true }).to_string().into_bytes())
                .with_header(content_type("application/json"))
                .with_status_code(409);
        }
        Err(e @ (recipes::SaveError::Invalid(_) | recipes::SaveError::Builtin(_))) => return json_err(400, &e.to_string()),
        Err(e @ recipes::SaveError::Io(_)) => return json_err(500, &e.to_string()),
    };
    println!("saved recipe {} to {}", saved.slug, recipes::my_recipes_path().unwrap_or_default().display());
    // Renders made with an older version of this recipe no longer match it.
    remove_renders(state, &saved.slug);
    let n = state.images.lock().unwrap().len();
    let image = body["image"].as_u64().map(|i| i as usize).filter(|&i| i < n);
    let from = body["from"].as_str().filter(|k| is_render_key(k));
    let mut linked = false;
    if let (Some(id), Some(from)) = (image, from) {
        let src = render_path(state, id, &format!("{from}.jpg"));
        linked = src.exists() && std::fs::copy(&src, render_path(state, id, &format!("{}.jpg", saved.slug))).is_ok();
    }
    let mut v = json!(saved);
    v["mine"] = json!(true);
    json_ok(json!({ "recipe": v, "linked": linked }))
}

fn delete_my_recipe(state: &State, slug: &str) -> Resp {
    match recipes::delete_mine(slug) {
        Ok(true) => {
            println!("deleted recipe {slug}");
            remove_renders(state, slug);
            json_ok(json!({ "ok": true }))
        }
        Ok(false) => json_err(404, "not one of your recipes"),
        Err(e) => json_err(500, &e.to_string()),
    }
}

/// Drop `<slug>.jpg` from every image's renders.
fn remove_renders(state: &State, slug: &str) {
    let n = state.images.lock().unwrap().len();
    for id in 0..n {
        let _ = std::fs::remove_file(render_path(state, id, &format!("{slug}.jpg")));
    }
}

fn read_json<T: serde::de::DeserializeOwned>(req: &mut Request) -> Result<T, String> {
    let mut data = Vec::new();
    req.as_reader()
        .take(MAX_JSON_BYTES + 1)
        .read_to_end(&mut data)
        .map_err(|e| format!("could not read request: {e}"))?;
    if data.len() as u64 > MAX_JSON_BYTES {
        return Err("request too large".into());
    }
    serde_json::from_slice(&data).map_err(|e| format!("invalid JSON: {e}"))
}

fn render_file(state: &State, id: usize, file: &str, download: bool) -> Resp {
    let Some(key) = file.strip_suffix(".jpg").filter(|s| is_render_key(s)) else {
        return json_err(404, "unknown render");
    };
    let path = render_path(state, id, &format!("{key}.jpg"));
    let disposition = download.then(|| format!("attachment; filename=\"{}.jpg\"", render_file_stem(state, id, key)));
    file_response(&path, "image/jpeg", disposition)
}

fn save_render(state: &State, id: usize, key: &str) -> Resp {
    if !is_render_key(key) {
        return json_err(404, "unknown recipe");
    }
    let src = render_path(state, id, &format!("{key}.jpg"));
    if !src.exists() {
        return json_err(404, "not converted yet");
    }
    let dest = state.root.join(format!("{}.jpg", render_file_stem(state, id, key)));
    match std::fs::copy(&src, &dest) {
        Ok(_) => {
            fuji::chown_to_sudo_user(&dest.to_string_lossy());
            println!("saved {}", dest.display());
            json_ok(json!({ "path": dest.display().to_string() }))
        }
        Err(e) => json_err(500, &format!("could not save: {e}")),
    }
}

// ---------------------------------------------------------------------------
// Matching a reference (film scan) — see `matcher`
// ---------------------------------------------------------------------------

/// Reference pictures larger than this are rejected (16-bit TIFF scans are big).
const MAX_REFERENCE_BYTES: u64 = 400 * 1024 * 1024;
/// Width of the aligned reference and the step previews shown in the viewer.
const MATCH_VIEW_W: u32 = 1600;

fn with_match(state: &State, job: &str, f: impl FnOnce(Arc<MatchJob>) -> Resp) -> Resp {
    match find_match(state, job) {
        Some(j) => f(j),
        None => json_err(404, "unknown match job"),
    }
}

fn start_match(state: &Arc<State>, id: usize, req: &mut Request, query: &str) -> Resp {
    let param = |k: &str| {
        query
            .split('&')
            .filter_map(|kv| kv.split_once('='))
            .find(|(key, _)| *key == k)
            .map(|(_, v)| percent_decode(v))
    };
    let max_renders = param("max").and_then(|v| v.parse::<usize>().ok()).unwrap_or(60).clamp(12, 200);
    // Start from a recipe slug, or "match:<job>" to continue from that run's best.
    let start = match param("start").filter(|s| !s.is_empty()) {
        None => None,
        Some(s) => match s.strip_prefix("match:") {
            Some(job) => match find_match(state, job).and_then(|j| j.status.lock().unwrap().best_params.clone()) {
                Some(p) => Some(Start::Params(p)),
                None => return json_err(404, "that match has no result to continue from"),
            },
            None if is_known_slug(&s) => Some(Start::Recipe(s)),
            None => return json_err(404, "unknown start recipe"),
        },
    };
    // The reference: uploaded in the body, or reused from an earlier run.
    let reuse = param("reference").and_then(|j| find_match(state, &j));
    let name = req
        .headers()
        .iter()
        .find(|h| h.field.equiv("X-Filename"))
        .map(|h| percent_decode(h.value.as_str()))
        .and_then(|n| Path::new(&n).file_name().map(|f| f.to_string_lossy().into_owned()))
        .unwrap_or_else(|| "reference".into());

    let mut data = Vec::new();
    if let Err(e) = req.as_reader().take(MAX_REFERENCE_BYTES + 1).read_to_end(&mut data) {
        return json_err(400, &format!("upload failed: {e}"));
    }
    if data.len() as u64 > MAX_REFERENCE_BYTES {
        return json_err(413, "file too large");
    }
    let name = match &reuse {
        Some(prev) if data.is_empty() => {
            data = match std::fs::read(prev.dir.join("source")) {
                Ok(d) => d,
                Err(e) => return json_err(410, &format!("the earlier reference is gone: {e}")),
            };
            prev.reference.clone()
        }
        _ => name,
    };
    let reference = match image::load_from_memory(&data) {
        Ok(img) => img.to_rgb8(),
        Err(e) => return json_err(400, &format!("cannot read {name} as an image (JPEG, PNG or TIFF): {e}")),
    };

    let job = {
        let mut jobs = state.matches.lock().unwrap();
        if jobs.iter().any(|j| matches!(j.status.lock().unwrap().state, "aligning" | "running")) {
            return json_err(409, "a match is already running — stop it first");
        }
        let jid = jobs.len();
        let dir = state.cache.join("matches").join(jid.to_string());
        if let Err(e) = std::fs::create_dir_all(&dir).and_then(|_| std::fs::write(dir.join("source"), &data)) {
            return json_err(500, &format!("cannot write to {}: {e}", dir.display()));
        }
        let job = Arc::new(MatchJob {
            id: jid,
            image: id,
            reference: name,
            dir,
            stop: AtomicBool::new(false),
            status: Mutex::new(MatchStatus {
                state: "aligning",
                phase: "Aligning the reference".into(),
                error: None,
                alignment: None,
                cells: 0,
                dropped: 0,
                monochrome: false,
                steps: Vec::new(),
                best: None,
                best_recipe: None,
                best_params: None,
                best_label: String::new(),
                max_renders,
                started: Instant::now(),
                finished: None,
                render_secs: Vec::new(),
                budget_exhausted: false,
                waiting_camera: None,
            }),
        });
        jobs.push(Arc::clone(&job));
        job
    };
    let (st, j) = (Arc::clone(state), Arc::clone(&job));
    std::thread::spawn(move || run_match(&st, &j, reference, start));
    json_ok(match_json(&job))
}

enum Start {
    Recipe(String),
    Params(matcher::Params),
}

fn find_match(state: &State, job: &str) -> Option<Arc<MatchJob>> {
    job.parse::<usize>().ok().and_then(|i| state.matches.lock().unwrap().get(i).cloned())
}

fn run_match(state: &State, job: &MatchJob, reference: image::RgbImage, start: Option<Start>) {
    let fail = |msg: String| {
        println!("  match failed: {msg}");
        let mut st = job.status.lock().unwrap();
        st.state = "error";
        st.error = Some(msg);
        st.finished = Some(st.started.elapsed().as_secs_f64());
    };
    let (name, path) = {
        let images = state.images.lock().unwrap();
        (images[job.image].name.clone(), images[job.image].path.clone())
    };
    let path_str = path.to_string_lossy().into_owned();
    println!("matching {name} against {} …", job.reference);

    // Every render shares the RAF's geometry, so align on the embedded preview.
    // Work in viewing orientation, so previews of vertical shots stand up.
    let crop = tele_crop(state, &path);
    let frame = std::fs::read(&path).ok().and_then(|raf| fuji::embedded_jpeg(&raf).and_then(|j| upright_preview(j, crop)));
    let Some(frame) = frame else {
        return fail("this RAF has no embedded preview to align the reference against".into());
    };
    let prepared = matcher::Reference::new(&reference, &frame);
    let a = prepared.alignment;
    println!(
        "  aligned: rot {}°, scale {:.2}, angle {:.1}°, score {:.2}, {} patches ({} dropped){}",
        a.rot90 as u32 * 90,
        a.scale,
        a.angle,
        a.score,
        prepared.cell_count(),
        prepared.dropped,
        if a.confident { "" } else { " (not confident)" }
    );
    let warped = matcher::warp_reference(&reference, &a, MATCH_VIEW_W);
    drop(reference);
    if let Err(e) = std::fs::write(job.dir.join("reference.jpg"), matcher::encode_jpeg(&warped, 88)) {
        return fail(format!("cannot write preview: {e}"));
    }
    {
        let mut st = job.status.lock().unwrap();
        st.alignment = Some(json!(a));
        st.cells = prepared.cell_count();
        st.dropped = prepared.dropped;
        st.monochrome = prepared.monochrome;
    }
    if prepared.cell_count() < 60 {
        return fail("the reference barely overlaps this photo — is it the same scene?".into());
    }

    let meta = fuji::read_raf_meta(&path_str);
    let mut dr_options: Vec<u32> = Vec::new();
    for dr in [100, 200, 400] {
        let d = fuji::clamp_dr(dr, meta.iso, meta.dynamic_range).dr;
        if !dr_options.contains(&d) {
            dr_options.push(d);
        }
    }
    let default_dr = fuji::clamp_dr(meta.dynamic_range.unwrap_or(100), meta.iso, meta.dynamic_range).dr;
    let start = match start {
        Some(Start::Params(p)) => Some(p),
        Some(Start::Recipe(slug)) => recipes::all()
            .iter()
            .find(|r| r.slug == slug)
            .map(|r| matcher::Params::from_settings(&r.to_settings(), default_dr)),
        None => None,
    }
    .map(|mut p| {
        p.dr = fuji::clamp_dr(p.dr, meta.iso, meta.dynamic_range).dr;
        p
    });
    let max_renders = job.status.lock().unwrap().max_renders;
    let opts = matcher::SearchOptions { max_renders, start, dr_options, default_dr, good_enough: 1.0 };
    {
        let mut st = job.status.lock().unwrap();
        st.state = "running";
        st.phase = "Starting".into();
    }

    let stub_frame = std::env::var("FXRAW_STUB_CAMERA").is_ok().then(|| matcher::thumbnail(&frame, MATCH_VIEW_W));
    let mut count = 0usize;
    let mut render = |settings: &crate::profile::RecipeSettings| -> Result<matcher::Rendered, String> {
        count += 1;
        let started = Instant::now();
        let out = if let Some(f) = &stub_frame {
            std::thread::sleep(std::time::Duration::from_millis(350));
            let image = matcher::simulate(f, settings);
            let jpeg = matcher::encode_jpeg(&image, 90);
            matcher::Rendered { image, jpeg }
        } else {
            // A long run can outlast the camera (battery, overheating, auto
            // power-off, a hung conversion): wait for it and retry the same
            // render instead of throwing the run away.
            let mut failures = 0;
            loop {
                let result = {
                    let _guard = state.camera.lock().unwrap();
                    *state.busy.lock().unwrap() = Some(format!("Matching {name} · render {count}/{max_renders}"));
                    let r = fuji::try_open_camera().and_then(|mut camera| {
                        fuji::convert_raf_to_jpeg(&mut *camera, &path_str, settings, &ui::ConvertProgress::quiet())
                    });
                    *state.busy.lock().unwrap() = None;
                    r
                };
                match result {
                    Ok((jpeg, _)) => {
                        job.status.lock().unwrap().waiting_camera = None;
                        let image = decode_upright(&jpeg).map_err(|e| format!("camera JPEG unreadable: {e}"))?;
                        break matcher::Rendered { image, jpeg };
                    }
                    Err(e) => {
                        failures += 1;
                        println!("  render {count} failed ({e}); waiting for the camera…");
                        if failures > 3 {
                            job.status.lock().unwrap().waiting_camera = None;
                            return Err(format!("the camera keeps failing: {e}"));
                        }
                        job.status.lock().unwrap().waiting_camera = Some(e.clone());
                        if !wait_for_camera(&job.stop, std::time::Duration::from_secs(600)) {
                            job.status.lock().unwrap().waiting_camera = None;
                            return Err(if job.stop.load(Ordering::Relaxed) {
                                "stopped while waiting for the camera".into()
                            } else {
                                format!("the camera didn't come back within 10 minutes ({e})")
                            });
                        }
                    }
                }
            }
        };
        job.status.lock().unwrap().render_secs.push(started.elapsed().as_secs_f64());
        Ok(out)
    };
    let ref_stem = Path::new(&job.reference).file_stem().unwrap_or_default().to_string_lossy().into_owned();
    let recipe_name = format!("Match · {ref_stem}");
    let recipe_slug = format!(
        "match-{}",
        ref_stem
            .to_lowercase()
            .chars()
            .map(|c| if c.is_ascii_alphanumeric() { c } else { '-' })
            .collect::<String>()
    );
    let recipe_json = |p: &matcher::Params| {
        let mut v = p.to_recipe_json(&recipe_name, &recipe_slug);
        v["source"] = json!({ "name": "fxraw match", "author": job.reference });
        v
    };
    let mut on_event = |e: matcher::Event| match e {
        matcher::Event::Phase(p) => job.status.lock().unwrap().phase = p.to_string(),
        matcher::Event::Evaluated { eval, rendered, is_best } => {
            let view = matcher::thumbnail(&rendered.image, MATCH_VIEW_W);
            let _ = std::fs::write(job.dir.join(format!("step-{}.jpg", eval.n)), matcher::encode_jpeg(&view, 85));
            if is_best {
                let _ = std::fs::write(job.dir.join("best-full.jpg"), &rendered.jpeg);
            }
            println!("  #{:<3} {:<16} {:<34} ΔE {:.2}{}", eval.n, eval.phase, eval.label, eval.m.loss, if is_best { "  ★" } else { "" });
            let mut st = job.status.lock().unwrap();
            st.steps.push(json!({
                "n": eval.n,
                "phase": eval.phase,
                "label": eval.label,
                "loss": round2(eval.m.loss),
                "mean_de": round2(eval.m.mean_de),
                "best": is_best,
                "stats": eval.m.stats,
                "recipe": recipe_json(&eval.params),
            }));
            if is_best {
                st.best = Some((eval.n, eval.m.loss));
                st.best_recipe = Some(recipe_json(&eval.params));
                st.best_params = Some(eval.params.clone());
                st.best_label = eval.label.clone();
            }
        }
    };
    let result = matcher::search(&prepared, &opts, &job.stop, &mut render, &mut on_event);

    let mut st = job.status.lock().unwrap();
    st.finished = Some(st.started.elapsed().as_secs_f64());
    match result {
        Ok(r) => {
            st.state = if r.stopped { "stopped" } else { "done" };
            st.budget_exhausted = r.budget_exhausted;
            st.error = r.error;
            st.phase = if r.stopped { "Stopped".into() } else { "Done".into() };
            println!("  match done: best ΔE {:.2} after {} renders", r.best.m.loss, r.renders);
        }
        Err(e) => {
            drop(st);
            fail(e);
        }
    }
}

/// Wait until the camera is (back) on USB and has settled. False when
/// stopped or timed out.
fn wait_for_camera(stop: &AtomicBool, timeout: std::time::Duration) -> bool {
    let started = Instant::now();
    let mut seen_since: Option<Instant> = None;
    while started.elapsed() < timeout {
        if stop.load(Ordering::Relaxed) {
            return false;
        }
        if camera_connected() {
            let since = *seen_since.get_or_insert_with(Instant::now);
            if since.elapsed() >= std::time::Duration::from_secs(4) {
                return true;
            }
        } else {
            seen_since = None;
        }
        std::thread::sleep(std::time::Duration::from_millis(1000));
    }
    false
}

fn round2(v: f32) -> f64 {
    (v as f64 * 100.0).round() / 100.0
}

fn match_json(job: &MatchJob) -> Value {
    let st = job.status.lock().unwrap();
    let elapsed = st.finished.unwrap_or_else(|| st.started.elapsed().as_secs_f64());
    let renders = st.steps.len();
    // ETA from the median render time and the renders left in the budget
    // (the search usually stops before using all of them).
    let eta = if matches!(st.state, "running") && !st.render_secs.is_empty() {
        let mut t = st.render_secs.clone();
        t.sort_by(f64::total_cmp);
        Some((t[t.len() / 2] * st.max_renders.saturating_sub(renders) as f64).round())
    } else {
        None
    };
    json!({
        "id": job.id,
        "image": job.image,
        "reference": job.reference,
        "state": st.state,
        "phase": st.phase,
        "error": st.error,
        "alignment": st.alignment,
        "cells": st.cells,
        "dropped": st.dropped,
        "waiting_camera": st.waiting_camera,
        "monochrome": st.monochrome,
        "renders": renders,
        "max_renders": st.max_renders,
        "budget_exhausted": st.budget_exhausted,
        "elapsed": elapsed.round(),
        "eta": eta,
        "start_loss": st.steps.first().map(|s| s["loss"].clone()),
        "best": st.best.map(|(n, loss)| json!({
            "n": n,
            "loss": round2(loss),
            "label": st.best_label,
            "recipe": st.best_recipe,
        })),
        "steps": st.steps,
    })
}

fn match_file(state: &State, job: &MatchJob, file: &str, download: bool) -> Resp {
    let path = match file {
        "reference.jpg" => job.dir.join("reference.jpg"),
        "best.jpg" => job.dir.join("best-full.jpg"),
        f => match f.strip_prefix("step-").and_then(|r| r.strip_suffix(".jpg")).and_then(|n| n.parse::<usize>().ok()) {
            Some(n) => job.dir.join(format!("step-{n}.jpg")),
            None => return json_err(404, "not found"),
        },
    };
    let disposition =
        download.then(|| format!("attachment; filename=\"{}-match.jpg\"", image_stem(state, job.image)));
    file_response(&path, "image/jpeg", disposition)
}

fn save_match(state: &State, job: &MatchJob) -> Resp {
    let recipe = job.status.lock().unwrap().best_recipe.clone();
    let Some(recipe) = recipe else {
        return json_err(404, "no render yet");
    };
    let stem = image_stem(state, job.image);
    let jpg = state.root.join(format!("{stem}-match.jpg"));
    let json_path = state.root.join(format!("{stem}-match.json"));
    let written = std::fs::copy(job.dir.join("best-full.jpg"), &jpg)
        .and_then(|_| std::fs::write(&json_path, serde_json::to_string_pretty(&recipe).unwrap_or_default() + "\n"));
    match written {
        Ok(()) => {
            for p in [&jpg, &json_path] {
                fuji::chown_to_sudo_user(&p.to_string_lossy());
            }
            println!("saved {} and {}", jpg.display(), json_path.display());
            json_ok(json!({ "jpg": jpg.display().to_string(), "json": json_path.display().to_string() }))
        }
        Err(e) => json_err(500, &format!("could not save: {e}")),
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn is_known_slug(slug: &str) -> bool {
    recipes::all().iter().any(|r| r.slug == slug)
}

fn image_stem(state: &State, id: usize) -> String {
    let images = state.images.lock().unwrap();
    Path::new(&images[id].name)
        .file_stem()
        .unwrap_or_default()
        .to_string_lossy()
        .into_owned()
}

fn camera_connected() -> bool {
    use rusb::UsbContext;
    if std::env::var("FXRAW_STUB_CAMERA").is_ok() {
        return true;
    }
    rusb::Context::new()
        .and_then(|ctx| ctx.devices())
        .map(|devs| {
            devs.iter().any(|d| {
                d.device_descriptor()
                    .is_ok_and(|desc| desc.vendor_id() == fuji::FUJIFILM_VENDOR_ID)
            })
        })
        .unwrap_or(false)
}

fn which(bin: &str) -> bool {
    std::process::Command::new(bin)
        .arg("-ver")
        .output()
        .is_ok_and(|o| o.status.success())
}

/// The EXIF TIFF block of a JPEG and its byte order (true = little endian).
fn exif_tiff(jpeg: &[u8]) -> Option<(&[u8], bool)> {
    // Walk JPEG markers to the APP1 "Exif" segment.
    let mut i = 2;
    while i + 4 <= jpeg.len() && jpeg[i] == 0xFF {
        let marker = jpeg[i + 1];
        let len = u16::from_be_bytes([jpeg[i + 2], jpeg[i + 3]]) as usize;
        let seg = jpeg.get(i + 4..i + 2 + len)?;
        if marker == 0xE1 && seg.starts_with(b"Exif\0\0") {
            let tiff = &seg[6..];
            let le = match tiff.get(..2)? {
                b"II" => true,
                b"MM" => false,
                _ => return None,
            };
            return Some((tiff, le));
        }
        if marker == 0xDA {
            break;
        }
        i += 2 + len;
    }
    None
}

struct Tiff<'a> {
    data: &'a [u8],
    le: bool,
}

impl Tiff<'_> {
    fn u16_at(&self, o: usize) -> Option<usize> {
        let b: [u8; 2] = self.data.get(o..o + 2)?.try_into().ok()?;
        Some(if self.le { u16::from_le_bytes(b) } else { u16::from_be_bytes(b) } as usize)
    }
    fn u32_at(&self, o: usize) -> Option<usize> {
        let b: [u8; 4] = self.data.get(o..o + 4)?.try_into().ok()?;
        Some(if self.le { u32::from_le_bytes(b) } else { u32::from_be_bytes(b) } as usize)
    }
    /// Offsets of the entries of the IFD at `ifd`: (tag, entry offset).
    fn entries(&self, ifd: usize) -> Option<Vec<(usize, usize)>> {
        let n = self.u16_at(ifd)?;
        (0..n).map(|k| Some((self.u16_at(ifd + 2 + k * 12)?, ifd + 2 + k * 12))).collect()
    }
}

/// Locate the EXIF (IFD1) thumbnail inside a JPEG.
fn exif_thumbnail(jpeg: &[u8]) -> Option<&[u8]> {
    let (data, le) = exif_tiff(jpeg)?;
    let t = Tiff { data, le };
    let ifd0 = t.u32_at(4)?;
    let ifd1 = t.u32_at(ifd0 + 2 + t.u16_at(ifd0)? * 12)?;
    if ifd1 == 0 {
        return None;
    }
    let (mut off, mut size) = (None, None);
    for (tag, e) in t.entries(ifd1)? {
        match tag {
            0x0201 => off = t.u32_at(e + 8),
            0x0202 => size = t.u32_at(e + 8),
            _ => {}
        }
    }
    let (off, size) = (off?, size?);
    let thumb = data.get(off..off + size)?;
    thumb.starts_with(&[0xFF, 0xD8]).then_some(thumb)
}

/// EXIF orientation of a JPEG (1 = upright, 6 = turn 90° clockwise to view,
/// 8 = 90° counter-clockwise, 3 = 180°).
pub(crate) fn exif_orientation(jpeg: &[u8]) -> u16 {
    let get = || -> Option<u16> {
        let (data, le) = exif_tiff(jpeg)?;
        let t = Tiff { data, le };
        let ifd0 = t.u32_at(4)?;
        let (_, e) = t.entries(ifd0)?.into_iter().find(|(tag, _)| *tag == 0x0112)?;
        Some(t.u16_at(e + 8)? as u16)
    };
    get().unwrap_or(1)
}

/// The digital tele-converter crop a RAF was shot with, as a normalised
/// (x, y, w, h) rectangle of the unrotated sensor frame. The camera applies
/// it when developing, while the embedded preview shows the whole frame.
fn tele_crop(state: &State, path: &Path) -> Option<[f32; 4]> {
    if let Some(c) = state.crops.lock().unwrap().get(path) {
        return *c;
    }
    let c = read_tele_crop(path);
    state.crops.lock().unwrap().insert(path.to_path_buf(), c);
    c
}

fn read_tele_crop(path: &Path) -> Option<[f32; 4]> {
    let out = std::process::Command::new("exiftool")
        .args(["-s", "-s", "-s", "-n", "-RAF:RawZoomActive", "-RAF:RawZoomTopLeft", "-RAF:RawZoomSize", "-RAF:RawImageCroppedSize"])
        .arg(path)
        .output()
        .ok()?;
    let rows: Vec<Vec<f32>> = String::from_utf8_lossy(&out.stdout)
        .lines()
        .map(|l| l.split_whitespace().filter_map(|x| x.parse().ok()).collect())
        .collect();
    parse_tele_crop(&rows)
}

/// `rows`: RawZoomActive, RawZoomTopLeft (x y), RawZoomSize (w h),
/// RawImageCroppedSize (w h), as printed by `exiftool -n`.
fn parse_tele_crop(rows: &[Vec<f32>]) -> Option<[f32; 4]> {
    let [active, tl, size, full] = rows else { return None };
    if active.first() != Some(&1.0) || tl.len() < 2 || size.len() < 2 || full.len() < 2 || full[0] <= 0.0 || full[1] <= 0.0 {
        return None;
    }
    let r = [tl[0] / full[0], tl[1] / full[1], size[0] / full[0], size[1] / full[1]];
    (r[2] > 0.1 && r[2] < 0.99 && r[3] > 0.1 && r[3] < 0.99 && r[0] + r[2] <= 1.001 && r[1] + r[3] <= 1.001).then_some(r)
}

/// The RAF's embedded preview as it should be viewed: tele-converter crop
/// (in sensor coordinates) applied first, then the EXIF orientation.
fn upright_preview(jpeg: &[u8], crop: Option<[f32; 4]>) -> Option<image::RgbImage> {
    let mut img = image::load_from_memory(jpeg).ok()?.to_rgb8();
    if let Some([x, y, w, h]) = crop {
        let (iw, ih) = (img.width() as f32, img.height() as f32);
        img = image::imageops::crop_imm(&img, (x * iw) as u32, (y * ih) as u32, (w * iw) as u32, (h * ih) as u32).to_image();
    }
    Some(rotate_by_orientation(img, exif_orientation(jpeg)))
}

fn rotate_by_orientation(img: image::RgbImage, orientation: u16) -> image::RgbImage {
    match orientation {
        3 => image::imageops::rotate180(&img),
        6 => image::imageops::rotate90(&img),
        8 => image::imageops::rotate270(&img),
        _ => img,
    }
}

/// Decode `jpeg` and turn it as EXIF `orientation` says.
fn decode_upright_as(jpeg: &[u8], orientation: u16) -> Option<image::RgbImage> {
    Some(rotate_by_orientation(image::load_from_memory(jpeg).ok()?.to_rgb8(), orientation))
}

/// Decode a JPEG into pixels the way it's meant to be viewed (EXIF
/// orientation applied; the `image` crate leaves that to the caller).
pub(crate) fn decode_upright(jpeg: &[u8]) -> Result<image::RgbImage, String> {
    decode_upright_as(jpeg, exif_orientation(jpeg)).ok_or_else(|| "not a readable JPEG".to_string())
}

fn percent_decode(s: &str) -> String {
    let bytes = s.as_bytes();
    let mut out = Vec::with_capacity(bytes.len());
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i] == b'%' && i + 2 < bytes.len() {
            let hex = std::str::from_utf8(&bytes[i + 1..i + 3]).unwrap_or("");
            if let Ok(v) = u8::from_str_radix(hex, 16) {
                out.push(v);
                i += 3;
                continue;
            }
        }
        out.push(bytes[i]);
        i += 1;
    }
    String::from_utf8_lossy(&out).into_owned()
}

fn content_type(v: &str) -> Header {
    Header::from_bytes("Content-Type", v).unwrap()
}

fn json_ok(v: Value) -> Resp {
    Response::from_data(v.to_string().into_bytes()).with_header(content_type("application/json"))
}

fn json_err(code: u16, msg: &str) -> Resp {
    Response::from_data(json!({ "error": msg }).to_string().into_bytes())
        .with_header(content_type("application/json"))
        .with_status_code(code)
}

fn file_response(path: &Path, mime: &str, disposition: Option<String>) -> Resp {
    match std::fs::read(path) {
        Ok(data) => {
            let mut r = Response::from_data(data)
                .with_header(content_type(mime))
                .with_header(Header::from_bytes("Cache-Control", "no-cache").unwrap());
            if let Some(d) = disposition {
                r = r.with_header(Header::from_bytes("Content-Disposition", d).unwrap());
            }
            r
        }
        Err(_) => json_err(404, "not found"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn extracts_embedded_preview_and_exif_thumbnail() {
        let raf = std::fs::read(concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures/DSCF6556.RAF")).unwrap();
        let full = fuji::embedded_jpeg(&raf).expect("embedded JPEG");
        assert!(full.len() > 1_000_000);
        let thumb = exif_thumbnail(full).expect("EXIF thumbnail");
        assert_eq!(exif_orientation(full), 1);
        assert_eq!(exif_orientation(thumb), 1); // no EXIF of its own
        assert!(thumb.starts_with(&[0xFF, 0xD8]) && thumb.len() < 100_000);
        assert!(fuji::embedded_jpeg(b"not a raf").is_none());
    }

    #[test]
    fn reads_tele_converter_crop() {
        // DSCF1593: digital tele-converter (≈1.4×), centred.
        let rows = vec![vec![1.0], vec![1128.0, 752.0], vec![5472.0, 3648.0], vec![7728.0, 5152.0]];
        let c = parse_tele_crop(&rows).unwrap();
        assert!((c[0] - 0.146).abs() < 0.001 && (c[1] - 0.146).abs() < 0.001);
        assert!((1.0 / c[2] - 1.41).abs() < 0.01 && (c[2] - c[3]).abs() < 0.001);
        // Not active, or tags missing (e.g. other cameras).
        let off = vec![vec![0.0], vec![0.0, 0.0], vec![7728.0, 5152.0], vec![7728.0, 5152.0]];
        assert!(parse_tele_crop(&off).is_none());
        assert!(parse_tele_crop(&[]).is_none());
    }

    #[test]
    fn percent_decodes_filenames() {
        assert_eq!(percent_decode("my%20photo.RAF"), "my photo.RAF");
        assert_eq!(percent_decode("100%"), "100%");
        assert_eq!(percent_decode("%E2%86%92"), "→");
    }
}
