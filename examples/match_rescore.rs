//! Diagnostics for `fxraw serve`'s reference matching: re-score renders
//! against a reference with the global alignment only and with per-patch
//! local alignment, and draw the per-patch error of the last render.
//!
//!     cargo run --release --example match_rescore -- REF.jpg PHOTO.RAF out.png RENDER.jpg...

use fxraw::{fuji, matcher};

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let [reference, raf, out, renders @ ..] = args.as_slice() else {
        eprintln!("usage: match_rescore REF.jpg PHOTO.RAF out.png RENDER.jpg...");
        std::process::exit(2);
    };
    let reference = image::open(reference).expect("reference").to_rgb8();
    let raf = std::fs::read(raf).expect("raf");
    let frame = image::load_from_memory(fuji::embedded_jpeg(&raf).expect("preview")).unwrap().to_rgb8();

    let t = std::time::Instant::now();
    let a = matcher::align(&reference, &frame);
    println!("align {:?} in {:.2}s", a, t.elapsed().as_secs_f32());
    let global = matcher::Reference::with_alignment(&reference, a);
    let t = std::time::Instant::now();
    let mut local = matcher::Reference::with_alignment(&reference, a);
    local.refine_locally(&reference, &frame);
    println!(
        "patches: global {}, local {} (+{} dropped) in {:.2}s",
        global.cell_count(),
        local.cell_count(),
        local.dropped,
        t.elapsed().as_secs_f32()
    );
    let mut last = None;
    for r in renders {
        let img = image::open(r).expect("render").to_rgb8();
        let (g, l) = (global.measure(&img), local.measure(&img));
        println!(
            "{:<14} global {:6.2}  local {:6.2}  patch {:5.2}  tone {:5.2}  mean {:5.2}  {:?}",
            r.rsplit('/').next().unwrap(),
            g.loss,
            l.loss,
            l.patch_de,
            l.tone_err,
            l.mean_de,
            l.stats
        );
        last = Some(img);
    }
    if let Some(img) = last {
        let mut view = matcher::thumbnail(&img, 900);
        let (w, h) = view.dimensions();
        for (x, y, de, _) in local.patch_errors(&img) {
            let t = (de / 15.0).clamp(0.0, 1.0);
            let col = image::Rgb([(255.0 * t) as u8, (255.0 * (1.0 - t)) as u8, 40]);
            let (cx, cy) = ((x * w as f32) as i32, (y * h as f32) as i32);
            for dy in -3..=3 {
                for dx in -3..=3 {
                    let (px, py) = (cx + dx, cy + dy);
                    if px >= 0 && py >= 0 && (px as u32) < w && (py as u32) < h {
                        view.put_pixel(px as u32, py as u32, col);
                    }
                }
            }
        }
        view.save(out).unwrap();
    }
}
