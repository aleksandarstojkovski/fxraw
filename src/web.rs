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
    monochrome: bool,
    steps: Vec<Value>,
    /// (render number, loss) of the best render so far.
    best: Option<(usize, f32)>,
    best_recipe: Option<Value>,
    best_label: String,
    max_renders: usize,
    started: Instant,
    finished: Option<f64>,
    /// Wall time of each camera render, for the ETA.
    render_secs: Vec<f64>,
    budget_exhausted: bool,
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
        (Method::Get, ["api", "recipes"]) => json_ok(json!(recipes::all())),
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
    let ids: Vec<usize> = found
        .into_iter()
        .map(|p| {
            let name = p.file_name().unwrap_or_default().to_string_lossy().into_owned();
            register(state, name, p, false)
        })
        .collect();
    json!(ids.into_iter().map(|id| image_json(state, id)).collect::<Vec<_>>())
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
    json!({ "id": id, "name": name, "path": rel, "uploaded": uploaded, "renders": renders })
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
        let bytes = if thumb { exif_thumbnail(full).unwrap_or(full) } else { full };
        if let Err(e) = std::fs::write(&cached, bytes) {
            return json_err(500, &format!("cache write failed: {e}"));
        }
    }
    file_response(&cached, "image/jpeg", None)
}

fn analysis(state: &State, id: usize) -> Resp {
    let path = state.images.lock().unwrap()[id].path.clone();
    match analyse::analyse_file(&path, 5) {
        Ok(a) => json_ok(json!({
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
    let Some(recipe) = recipes::all().iter().find(|r| r.slug == slug) else {
        return json_err(404, "unknown recipe");
    };
    let out_dir = state.cache.join("renders").join(id.to_string());
    let out = out_dir.join(format!("{slug}.jpg"));
    if out.exists() {
        return json_ok(json!({ "slug": slug, "cached": true, "warnings": [] }));
    }
    let (name, path) = {
        let images = state.images.lock().unwrap();
        (images[id].name.clone(), images[id].path.clone())
    };

    let _guard = state.camera.lock().unwrap();
    // A queued duplicate request may have produced it while we waited.
    if out.exists() {
        return json_ok(json!({ "slug": slug, "cached": true, "warnings": [] }));
    }
    *state.busy.lock().unwrap() = Some(format!("{name} → {}", recipe.name));
    println!("converting {name} → {} …", recipe.slug);
    let started = Instant::now();

    let result = fuji::try_open_camera().and_then(|mut camera| {
        fuji::convert_raf_to_jpeg(
            &mut *camera,
            &path.to_string_lossy(),
            &recipe.to_settings(),
            &ui::ConvertProgress::quiet(),
        )
    });
    *state.busy.lock().unwrap() = None;
    let secs = started.elapsed().as_secs_f64();

    match result {
        Ok((jpeg, warnings)) => {
            if let Err(e) = std::fs::create_dir_all(&out_dir).and_then(|_| std::fs::write(&out, &jpeg)) {
                return json_err(500, &format!("could not store render: {e}"));
            }
            println!("  done in {secs:.1}s ({:.1} MB)", jpeg.len() as f64 / 1_048_576.0);
            for w in &warnings {
                println!("  {w}");
            }
            json_ok(json!({ "slug": slug, "cached": false, "seconds": secs, "warnings": warnings }))
        }
        Err(e) => {
            println!("  failed: {e}");
            json_err(502, &e)
        }
    }
}

fn render_file(state: &State, id: usize, file: &str, download: bool) -> Resp {
    let Some(slug) = file.strip_suffix(".jpg").filter(|s| is_known_slug(s)) else {
        return json_err(404, "unknown render");
    };
    let path = state.cache.join("renders").join(id.to_string()).join(format!("{slug}.jpg"));
    let disposition = download.then(|| {
        let stem = image_stem(state, id);
        format!("attachment; filename=\"{stem}-{slug}.jpg\"")
    });
    file_response(&path, "image/jpeg", disposition)
}

fn save_render(state: &State, id: usize, slug: &str) -> Resp {
    if !is_known_slug(slug) {
        return json_err(404, "unknown recipe");
    }
    let src = state.cache.join("renders").join(id.to_string()).join(format!("{slug}.jpg"));
    if !src.exists() {
        return json_err(404, "not converted yet");
    }
    let dest = state.root.join(format!("{}-{slug}.jpg", image_stem(state, id)));
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
    let job = job.parse::<usize>().ok().and_then(|i| state.matches.lock().unwrap().get(i).cloned());
    match job {
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
    let start = param("start").filter(|s| !s.is_empty());
    if let Some(slug) = &start
        && !is_known_slug(slug)
    {
        return json_err(404, "unknown start recipe");
    }
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
    let reference = match image::load_from_memory(&data) {
        Ok(img) => img.to_rgb8(),
        Err(e) => return json_err(400, &format!("cannot read {name} as an image (JPEG, PNG or TIFF): {e}")),
    };
    drop(data);

    let job = {
        let mut jobs = state.matches.lock().unwrap();
        if jobs.iter().any(|j| matches!(j.status.lock().unwrap().state, "aligning" | "running")) {
            return json_err(409, "a match is already running — stop it first");
        }
        let jid = jobs.len();
        let dir = state.cache.join("matches").join(jid.to_string());
        if let Err(e) = std::fs::create_dir_all(&dir) {
            return json_err(500, &format!("cannot create {}: {e}", dir.display()));
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
                monochrome: false,
                steps: Vec::new(),
                best: None,
                best_recipe: None,
                best_label: String::new(),
                max_renders,
                started: Instant::now(),
                finished: None,
                render_secs: Vec::new(),
                budget_exhausted: false,
            }),
        });
        jobs.push(Arc::clone(&job));
        job
    };
    let (st, j) = (Arc::clone(state), Arc::clone(&job));
    std::thread::spawn(move || run_match(&st, &j, reference, start));
    json_ok(match_json(&job))
}

fn run_match(state: &State, job: &MatchJob, reference: image::RgbImage, start: Option<String>) {
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
    let frame = std::fs::read(&path)
        .ok()
        .and_then(|raf| fuji::embedded_jpeg(&raf).and_then(|j| image::load_from_memory(j).ok()))
        .map(|i| i.to_rgb8());
    let Some(frame) = frame else {
        return fail("this RAF has no embedded preview to align the reference against".into());
    };
    let prepared = matcher::Reference::new(&reference, &frame);
    let a = prepared.alignment;
    println!(
        "  aligned: rot {}°, scale {:.2}, angle {:.1}°, score {:.2}, {} patches{}",
        a.rot90 as u32 * 90,
        a.scale,
        a.angle,
        a.score,
        prepared.cell_count(),
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
    let start = start.and_then(|slug| recipes::all().iter().find(|r| r.slug == slug)).map(|r| {
        let mut p = matcher::Params::from_settings(&r.to_settings(), default_dr);
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
            let _guard = state.camera.lock().unwrap();
            *state.busy.lock().unwrap() = Some(format!("Matching {name} · render {count}/{max_renders}"));
            let result = fuji::try_open_camera().and_then(|mut camera| {
                fuji::convert_raf_to_jpeg(&mut *camera, &path_str, settings, &ui::ConvertProgress::quiet())
            });
            *state.busy.lock().unwrap() = None;
            let (jpeg, _) = result?;
            let image = image::load_from_memory(&jpeg).map_err(|e| format!("camera JPEG unreadable: {e}"))?.to_rgb8();
            matcher::Rendered { image, jpeg }
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

/// Locate the EXIF (IFD1) thumbnail inside a JPEG.
fn exif_thumbnail(jpeg: &[u8]) -> Option<&[u8]> {
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
            let u16_at = |o: usize| -> Option<usize> {
                let b: [u8; 2] = tiff.get(o..o + 2)?.try_into().ok()?;
                Some(if le { u16::from_le_bytes(b) } else { u16::from_be_bytes(b) } as usize)
            };
            let u32_at = |o: usize| -> Option<usize> {
                let b: [u8; 4] = tiff.get(o..o + 4)?.try_into().ok()?;
                Some(if le { u32::from_le_bytes(b) } else { u32::from_be_bytes(b) } as usize)
            };
            let ifd0 = u32_at(4)?;
            let ifd1 = u32_at(ifd0 + 2 + u16_at(ifd0)? * 12)?;
            if ifd1 == 0 {
                return None;
            }
            let (mut off, mut size) = (None, None);
            for n in 0..u16_at(ifd1)? {
                let e = ifd1 + 2 + n * 12;
                match u16_at(e)? {
                    0x0201 => off = u32_at(e + 8),
                    0x0202 => size = u32_at(e + 8),
                    _ => {}
                }
            }
            let (off, size) = (off?, size?);
            let thumb = tiff.get(off..off + size)?;
            return thumb.starts_with(&[0xFF, 0xD8]).then_some(thumb);
        }
        if marker == 0xDA {
            break;
        }
        i += 2 + len;
    }
    None
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
        assert!(thumb.starts_with(&[0xFF, 0xD8]) && thumb.len() < 100_000);
        assert!(fuji::embedded_jpeg(b"not a raf").is_none());
    }

    #[test]
    fn percent_decodes_filenames() {
        assert_eq!(percent_decode("my%20photo.RAF"), "my photo.RAF");
        assert_eq!(percent_decode("100%"), "100%");
        assert_eq!(percent_decode("%E2%86%92"), "→");
    }
}
