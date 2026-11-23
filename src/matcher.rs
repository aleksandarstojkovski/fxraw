//! Recipe matching: search the recipe parameters for the settings whose
//! camera render looks most like a reference picture of the same scene
//! (typically a film scan shot alongside the Fuji).
//!
//! 1. **Align** — find the transform (90° turns, scale, small rotation,
//!    offset) that puts the reference on top of the RAF's framing, by
//!    normalised cross-correlation of luminance gradients (robust to the two
//!    images having different colours and tone curves).
//! 2. **Compare** — a grid of blurred patches in CIE Lab. The error is a
//!    trimmed, weighted mean of ΔE2000 (busy/clipped patches count less, the
//!    worst 10% — people that moved, parallax — are ignored). Signed
//!    differences (lighter/darker, warmer/cooler, more/less saturated)
//!    tell the search which way to turn each knob.
//! 3. **Search** — try every film simulation, then for each numeric knob
//!    measure the effect of one step and jump to the value that should
//!    cancel the difference (secant step), then hill-climb locally.
//!    Categorical knobs (Color Chrome, FX Blue, DR, WB mode) are tried.
//!    A change is kept only when the total error goes down.

use crate::profile::{ChromeLevel, FilmSimulation, GrainEffect, GrainSize, RecipeSettings};
use image::RgbImage;
use serde::Serialize;
use serde_json::{Value, json};
use std::collections::HashMap;
use std::sync::LazyLock;
use std::sync::atomic::{AtomicBool, Ordering};

// ---------------------------------------------------------------------------
// Colour
// ---------------------------------------------------------------------------

static SRGB_TO_LINEAR: LazyLock<[f32; 256]> = LazyLock::new(|| {
    let mut t = [0f32; 256];
    for (i, v) in t.iter_mut().enumerate() {
        let c = i as f32 / 255.0;
        *v = if c <= 0.04045 { c / 12.92 } else { ((c + 0.055) / 1.055).powf(2.4) };
    }
    t
});

fn lab_f(t: f32) -> f32 {
    if t > 0.008_856 { t.cbrt() } else { 7.787 * t + 16.0 / 116.0 }
}

/// Linear sRGB (D65) → CIE Lab.
pub fn linear_to_lab(p: [f32; 3]) -> [f32; 3] {
    let [r, g, b] = p;
    let x = 0.412_456_4 * r + 0.357_576_1 * g + 0.180_437_5 * b;
    let y = 0.212_672_9 * r + 0.715_152_2 * g + 0.072_175 * b;
    let z = 0.019_333_9 * r + 0.119_192 * g + 0.950_304_1 * b;
    let (fx, fy, fz) = (lab_f(x / 0.950_47), lab_f(y), lab_f(z / 1.088_83));
    [116.0 * fy - 16.0, 500.0 * (fx - fy), 200.0 * (fy - fz)]
}

fn chroma(l: &[f32; 3]) -> f32 {
    l[1].hypot(l[2])
}

/// CIEDE2000 colour difference.
pub fn delta_e2000(c1: [f32; 3], c2: [f32; 3]) -> f32 {
    let (l1, a1, b1) = (c1[0] as f64, c1[1] as f64, c1[2] as f64);
    let (l2, a2, b2) = (c2[0] as f64, c2[1] as f64, c2[2] as f64);
    let p7 = |x: f64| x.powi(7);
    let k = p7(25.0);
    let cbar = (a1.hypot(b1) + a2.hypot(b2)) / 2.0;
    let g = 0.5 * (1.0 - (p7(cbar) / (p7(cbar) + k)).sqrt());
    let (a1p, a2p) = ((1.0 + g) * a1, (1.0 + g) * a2);
    let (c1p, c2p) = (a1p.hypot(b1), a2p.hypot(b2));
    let hue = |b: f64, a: f64| {
        if a == 0.0 && b == 0.0 {
            0.0
        } else {
            let h = b.atan2(a).to_degrees();
            if h < 0.0 { h + 360.0 } else { h }
        }
    };
    let (h1p, h2p) = (hue(b1, a1p), hue(b2, a2p));
    let dlp = l2 - l1;
    let dcp = c2p - c1p;
    let dhp = if c1p * c2p == 0.0 {
        0.0
    } else {
        let d = h2p - h1p;
        if d > 180.0 {
            d - 360.0
        } else if d < -180.0 {
            d + 360.0
        } else {
            d
        }
    };
    let dhp_big = 2.0 * (c1p * c2p).sqrt() * (dhp.to_radians() / 2.0).sin();
    let lbp = (l1 + l2) / 2.0;
    let cbp = (c1p + c2p) / 2.0;
    let hbp = if c1p * c2p == 0.0 {
        h1p + h2p
    } else if (h1p - h2p).abs() <= 180.0 {
        (h1p + h2p) / 2.0
    } else if h1p + h2p < 360.0 {
        (h1p + h2p + 360.0) / 2.0
    } else {
        (h1p + h2p - 360.0) / 2.0
    };
    let cosd = |d: f64| d.to_radians().cos();
    let t = 1.0 - 0.17 * cosd(hbp - 30.0) + 0.24 * cosd(2.0 * hbp) + 0.32 * cosd(3.0 * hbp + 6.0)
        - 0.20 * cosd(4.0 * hbp - 63.0);
    let dtheta = 30.0 * (-((hbp - 275.0) / 25.0).powi(2)).exp();
    let rc = 2.0 * (p7(cbp) / (p7(cbp) + k)).sqrt();
    let sl = 1.0 + 0.015 * (lbp - 50.0).powi(2) / (20.0 + (lbp - 50.0).powi(2)).sqrt();
    let sc = 1.0 + 0.045 * cbp;
    let sh = 1.0 + 0.015 * cbp * t;
    let rt = -(2.0 * dtheta).to_radians().sin() * rc;
    let (x, y, z) = (dlp / sl, dcp / sc, dhp_big / sh);
    (x * x + y * y + z * z + rt * y * z).max(0.0).sqrt() as f32
}

// ---------------------------------------------------------------------------
// Working images (small, linear RGB)
// ---------------------------------------------------------------------------

#[derive(Clone)]
struct Img {
    w: usize,
    h: usize,
    px: Vec<[f32; 3]>,
}

impl Img {
    /// Area-average downscale to at most `max_w` pixels wide, in linear light.
    fn from_rgb(img: &RgbImage, max_w: usize) -> Img {
        let (sw, sh) = (img.width() as usize, img.height() as usize);
        let w = max_w.min(sw).max(1);
        let h = ((sh as f64 * w as f64 / sw as f64).round() as usize).max(1);
        let lut = &*SRGB_TO_LINEAR;
        let mut acc = vec![[0f32; 4]; w * h];
        let xmap: Vec<usize> = (0..sw).map(|x| x * w / sw).collect();
        let raw = img.as_raw();
        for y in 0..sh {
            let ty = y * h / sh;
            let row = &raw[y * sw * 3..(y + 1) * sw * 3];
            let out = &mut acc[ty * w..(ty + 1) * w];
            for (x, p) in row.chunks_exact(3).enumerate() {
                let a = &mut out[xmap[x]];
                a[0] += lut[p[0] as usize];
                a[1] += lut[p[1] as usize];
                a[2] += lut[p[2] as usize];
                a[3] += 1.0;
            }
        }
        let px = acc
            .into_iter()
            .map(|a| {
                let n = a[3].max(1.0);
                [a[0] / n, a[1] / n, a[2] / n]
            })
            .collect();
        Img { w, h, px }
    }

    /// Rotate by `k` × 90° clockwise.
    fn rotate(&self, k: u8) -> Img {
        let k = k % 4;
        if k == 0 {
            return self.clone();
        }
        let (w, h) = if k % 2 == 1 { (self.h, self.w) } else { (self.w, self.h) };
        let mut px = Vec::with_capacity(w * h);
        for y in 0..h {
            for x in 0..w {
                let (ox, oy) = match k {
                    1 => (y, self.h - 1 - x),
                    2 => (self.w - 1 - x, self.h - 1 - y),
                    _ => (self.w - 1 - y, x),
                };
                px.push(self.px[oy * self.w + ox]);
            }
        }
        Img { w, h, px }
    }

    fn blur(&self, r: usize) -> Img {
        let mut chans: Vec<Vec<f32>> = (0..3).map(|c| self.px.iter().map(|p| p[c]).collect()).collect();
        for ch in chans.iter_mut() {
            *ch = box_blur(ch, self.w, self.h, r);
        }
        let px = (0..self.px.len()).map(|i| [chans[0][i], chans[1][i], chans[2][i]]).collect();
        Img { w: self.w, h: self.h, px }
    }

    /// Bilinear sample at normalised coordinates (0..1 across the image).
    fn sample(&self, u: f32, v: f32) -> [f32; 3] {
        let x = (u * self.w as f32 - 0.5).clamp(0.0, (self.w - 1) as f32);
        let y = (v * self.h as f32 - 0.5).clamp(0.0, (self.h - 1) as f32);
        let (x0, y0) = (x as usize, y as usize);
        let (x1, y1) = ((x0 + 1).min(self.w - 1), (y0 + 1).min(self.h - 1));
        let (fx, fy) = (x - x0 as f32, y - y0 as f32);
        let p = |x, y| self.px[y * self.w + x];
        let mut out = [0f32; 3];
        for (c, o) in out.iter_mut().enumerate() {
            let top = p(x0, y0)[c] * (1.0 - fx) + p(x1, y0)[c] * fx;
            let bot = p(x0, y1)[c] * (1.0 - fx) + p(x1, y1)[c] * fx;
            *o = top * (1.0 - fy) + bot * fy;
        }
        out
    }

    /// CIE L* per pixel.
    fn lightness(&self) -> Gray {
        let v = self
            .px
            .iter()
            .map(|p| 116.0 * lab_f(0.212_672_9 * p[0] + 0.715_152_2 * p[1] + 0.072_175 * p[2]) - 16.0)
            .collect();
        Gray { w: self.w, h: self.h, v }
    }
}

struct Gray {
    w: usize,
    h: usize,
    v: Vec<f32>,
}

impl Gray {
    /// Gradient magnitude of the lightly blurred image — the alignment feature.
    fn gradient(&self) -> Gray {
        let s = box_blur(&self.v, self.w, self.h, 1);
        let at = |x: usize, y: usize| s[y * self.w + x];
        let mut v = vec![0f32; self.w * self.h];
        for y in 0..self.h {
            for x in 0..self.w {
                let gx = at((x + 1).min(self.w - 1), y) - at(x.saturating_sub(1), y);
                let gy = at(x, (y + 1).min(self.h - 1)) - at(x, y.saturating_sub(1));
                v[y * self.w + x] = gx.hypot(gy);
            }
        }
        Gray { w: self.w, h: self.h, v }
    }

    /// Bilinear sample at pixel coordinates; `None` outside the image.
    fn sample_px(&self, x: f32, y: f32) -> Option<f32> {
        if x < 0.0 || y < 0.0 || x > (self.w - 1) as f32 || y > (self.h - 1) as f32 {
            return None;
        }
        let (x0, y0) = (x as usize, y as usize);
        let (x1, y1) = ((x0 + 1).min(self.w - 1), (y0 + 1).min(self.h - 1));
        let (fx, fy) = (x - x0 as f32, y - y0 as f32);
        let p = |x, y| self.v[y * self.w + x];
        let top = p(x0, y0) * (1.0 - fx) + p(x1, y0) * fx;
        let bot = p(x0, y1) * (1.0 - fx) + p(x1, y1) * fx;
        Some(top * (1.0 - fy) + bot * fy)
    }
}

/// Separable box blur of radius `r` (window 2r+1), edges clamped.
fn box_blur(src: &[f32], w: usize, h: usize, r: usize) -> Vec<f32> {
    if r == 0 {
        return src.to_vec();
    }
    let pass = |src: &[f32], len: usize, count: usize, idx: &dyn Fn(usize, usize) -> usize| -> Vec<f32> {
        let mut out = vec![0f32; src.len()];
        let n = (2 * r + 1) as f32;
        for line in 0..count {
            let get = |i: isize| src[idx(line, i.clamp(0, len as isize - 1) as usize)];
            let mut sum: f32 = (-(r as isize)..=r as isize).map(get).sum();
            for i in 0..len {
                out[idx(line, i)] = sum / n;
                sum += get(i as isize + r as isize + 1) - get(i as isize - r as isize);
            }
        }
        out
    };
    let horiz = pass(src, w, h, &|line, i| line * w + i);
    pass(&horiz, h, w, &|line, i| i * w + line)
}

// ---------------------------------------------------------------------------
// Alignment
// ---------------------------------------------------------------------------

/// Where the reference sits on the render. Coordinates are normalised: the
/// reference (after `rot90` clockwise quarter turns) spans u,v ∈ 0..1, the
/// render spans x,y ∈ 0..1.
#[derive(Clone, Copy, Debug, Serialize)]
pub struct Alignment {
    pub rot90: u8,
    /// Reference width as a fraction of the render width.
    pub scale: f32,
    /// Extra rotation in degrees.
    pub angle: f32,
    /// Reference centre on the render.
    pub cx: f32,
    pub cy: f32,
    /// Normalised cross-correlation of the gradients (≈0.5+ is a solid lock).
    pub score: f32,
    /// Fraction of the reference that lands inside the render.
    pub overlap: f32,
    /// False when no convincing match was found and a plain fit was used.
    pub confident: bool,
    /// Reference aspect (h/w, after rot90) and render aspect (h/w).
    #[serde(skip)]
    ref_aspect: f32,
    #[serde(skip)]
    render_aspect: f32,
}

impl Alignment {
    /// Reference point (u,v) → render point (x,y).
    fn map(&self, u: f32, v: f32) -> (f32, f32) {
        let dx = (u - 0.5) * self.scale;
        let dy = (v - 0.5) * self.scale * self.ref_aspect;
        let (s, c) = self.angle.to_radians().sin_cos();
        let (rx, ry) = (dx * c - dy * s, dx * s + dy * c);
        (self.cx + rx, self.cy + ry / self.render_aspect)
    }

    /// Render point (x,y) → reference point (u,v).
    fn unmap(&self, x: f32, y: f32) -> (f32, f32) {
        let (rx, ry) = (x - self.cx, (y - self.cy) * self.render_aspect);
        let (s, c) = self.angle.to_radians().sin_cos();
        let (dx, dy) = (rx * c + ry * s, -rx * s + ry * c);
        (dx / self.scale + 0.5, dy / (self.scale * self.ref_aspect) + 0.5)
    }
}

/// Rotated-frame (u,v) → original reference (u,v), for `k` clockwise turns.
fn unrotate(k: u8, u: f32, v: f32) -> (f32, f32) {
    match k % 4 {
        0 => (u, v),
        1 => (v, 1.0 - u),
        2 => (1.0 - u, 1.0 - v),
        _ => (1.0 - v, u),
    }
}

/// Bounds (u0,v0,u1,v1) of the useful area of the reference: film borders,
/// scanner bed and similar flat frames around the picture are trimmed.
fn detect_crop(img: &Img) -> [f32; 4] {
    let g = img.lightness();
    let flat = |vals: &mut dyn Iterator<Item = f32>| {
        let v: Vec<f32> = vals.collect();
        let n = v.len() as f32;
        let mean = v.iter().sum::<f32>() / n;
        let sd = (v.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / n).sqrt();
        sd < 5.0 && (mean < 22.0 || mean > 90.0)
    };
    let max_x = g.w / 4;
    let max_y = g.h / 4;
    let mut l = 0;
    while l < max_x && flat(&mut (0..g.h).map(|y| g.v[y * g.w + l])) {
        l += 1;
    }
    let mut r = 0;
    while r < max_x && flat(&mut (0..g.h).map(|y| g.v[y * g.w + g.w - 1 - r])) {
        r += 1;
    }
    let mut t = 0;
    while t < max_y && flat(&mut (0..g.w).map(|x| g.v[t * g.w + x])) {
        t += 1;
    }
    let mut b = 0;
    while b < max_y && flat(&mut (0..g.w).map(|x| g.v[(g.h - 1 - b) * g.w + x])) {
        b += 1;
    }
    let inset = 0.015;
    [
        l as f32 / g.w as f32 + inset,
        t as f32 / g.h as f32 + inset,
        1.0 - r as f32 / g.w as f32 - inset,
        1.0 - b as f32 / g.h as f32 - inset,
    ]
}

/// The crop rectangle expressed in the frame rotated by `k` clockwise turns.
fn rotate_crop(k: u8, c: [f32; 4]) -> [f32; 4] {
    let [l, t, r, b] = c;
    match k % 4 {
        0 => c,
        1 => [1.0 - b, l, 1.0 - t, r],
        2 => [1.0 - r, 1.0 - b, 1.0 - l, 1.0 - t],
        _ => [t, 1.0 - r, b, 1.0 - l],
    }
}

#[derive(Clone, Copy)]
struct Pose {
    log_scale: f32,
    angle: f32,
    cx: f32,
    cy: f32,
}

struct AlignLevel {
    /// Reference gradient samples: normalised (u,v) and value.
    pts: Vec<(f32, f32, f32)>,
    render: Gray,
}

impl AlignLevel {
    fn new(reference: &Img, k: u8, crop: [f32; 4], render: &Img) -> AlignLevel {
        let r = reference.rotate(k);
        let g = r.lightness().gradient();
        let c = rotate_crop(k, crop);
        let mut pts = Vec::new();
        for y in 0..g.h {
            for x in 0..g.w {
                let (u, v) = ((x as f32 + 0.5) / g.w as f32, (y as f32 + 0.5) / g.h as f32);
                if u >= c[0] && u <= c[2] && v >= c[1] && v <= c[3] {
                    pts.push((u, v, g.v[y * g.w + x]));
                }
            }
        }
        AlignLevel { pts, render: render.lightness().gradient() }
    }

    fn score(&self, a: &Alignment) -> (f32, f32) {
        let (rw, rh) = (self.render.w as f32, self.render.h as f32);
        let (mut n, mut sa, mut sb, mut saa, mut sbb, mut sab) = (0f64, 0f64, 0f64, 0f64, 0f64, 0f64);
        for &(u, v, a_val) in &self.pts {
            let (x, y) = a.map(u, v);
            if let Some(b_val) = self.render.sample_px(x * rw - 0.5, y * rh - 0.5) {
                let (a_val, b_val) = (a_val as f64, b_val as f64);
                n += 1.0;
                sa += a_val;
                sb += b_val;
                saa += a_val * a_val;
                sbb += b_val * b_val;
                sab += a_val * b_val;
            }
        }
        let overlap = (n / self.pts.len().max(1) as f64) as f32;
        if overlap < 0.35 {
            return (-1.0, overlap);
        }
        let cov = sab - sa * sb / n;
        let va = saa - sa * sa / n;
        let vb = sbb - sb * sb / n;
        if va <= 1e-9 || vb <= 1e-9 {
            return (-1.0, overlap);
        }
        let ncc = (cov / (va * vb).sqrt()) as f32;
        // Prefer covering more of the reference when the correlation is equal.
        (ncc * (0.8 + 0.2 * overlap), overlap)
    }
}

fn pose_alignment(p: Pose, k: u8, ref_aspect: f32, render_aspect: f32) -> Alignment {
    Alignment {
        rot90: k,
        scale: p.log_scale.exp(),
        angle: p.angle,
        cx: p.cx,
        cy: p.cy,
        score: 0.0,
        overlap: 0.0,
        confident: true,
        ref_aspect,
        render_aspect,
    }
}

/// Find where `reference` sits on `frame` (any rendering of the RAF — the
/// embedded preview works, since every render has the same geometry).
pub fn align(reference: &RgbImage, frame: &RgbImage) -> Alignment {
    let ref_small = Img::from_rgb(reference, 48);
    let ref_fine = Img::from_rgb(reference, 160);
    let crop = detect_crop(&Img::from_rgb(reference, 256));
    let frame_small = Img::from_rgb(frame, 72);
    let frame_fine = Img::from_rgb(frame, 240);
    let render_aspect = frame.height() as f32 / frame.width() as f32;
    let ref_aspect0 = reference.height() as f32 / reference.width() as f32;

    // Coarse exhaustive search.
    let mut coarse: Vec<(f32, u8, Pose)> = Vec::new();
    for k in 0..4u8 {
        let ref_aspect = if k % 2 == 1 { 1.0 / ref_aspect0 } else { ref_aspect0 };
        let level = AlignLevel::new(&ref_small, k, crop, &frame_small);
        for si in 0..13 {
            let log_scale = (0.45f32).ln() + si as f32 * ((1.35f32 / 0.45).ln() / 12.0);
            for angle in [-2.0f32, 0.0, 2.0] {
                for yi in 0..21 {
                    for xi in 0..21 {
                        let p = Pose {
                            log_scale,
                            angle,
                            cx: 0.15 + xi as f32 * 0.035,
                            cy: 0.15 + yi as f32 * 0.035,
                        };
                        let (s, _) = level.score(&pose_alignment(p, k, ref_aspect, render_aspect));
                        if s > 0.0 {
                            coarse.push((s, k, p));
                        }
                    }
                }
            }
        }
    }
    coarse.sort_by(|a, b| b.0.total_cmp(&a.0));

    // Refine the best few distinct candidates on finer images.
    let mut best: Option<(f32, Alignment)> = None;
    let mut fine_levels: HashMap<u8, AlignLevel> = HashMap::new();
    let mut seeds: Vec<(u8, Pose)> = Vec::new();
    for &(_, k, p) in &coarse {
        let distinct = seeds.iter().all(|(sk, sp)| {
            *sk != k
                || (sp.log_scale - p.log_scale).abs() > 0.12
                || (sp.cx - p.cx).abs() > 0.08
                || (sp.cy - p.cy).abs() > 0.08
        });
        if distinct {
            seeds.push((k, p));
        }
        if seeds.len() == 6 {
            break;
        }
    }
    for (k, mut p) in seeds {
        let ref_aspect = if k % 2 == 1 { 1.0 / ref_aspect0 } else { ref_aspect0 };
        let level = fine_levels
            .entry(k)
            .or_insert_with(|| AlignLevel::new(&ref_fine, k, crop, &frame_fine));
        let eval = |p: Pose| level.score(&pose_alignment(p, k, ref_aspect, render_aspect)).0;
        let mut cur = eval(p);
        let mut steps = [0.05f32, 1.0, 0.02, 0.02];
        for _ in 0..7 {
            let mut moved = true;
            let mut guard = 0;
            while moved && guard < 20 {
                moved = false;
                guard += 1;
                for dim in 0..4 {
                    for sign in [-1.0f32, 1.0] {
                        let mut q = p;
                        match dim {
                            0 => q.log_scale += sign * steps[0],
                            1 => q.angle = (q.angle + sign * steps[1]).clamp(-6.0, 6.0),
                            2 => q.cx += sign * steps[2],
                            _ => q.cy += sign * steps[3],
                        }
                        let s = eval(q);
                        if s > cur {
                            cur = s;
                            p = q;
                            moved = true;
                        }
                    }
                }
            }
            for s in steps.iter_mut() {
                *s /= 2.0;
            }
        }
        if best.as_ref().is_none_or(|(b, _)| cur > *b) {
            let mut a = pose_alignment(p, k, ref_aspect, render_aspect);
            let (score, overlap) = level.score(&a);
            a.score = score;
            a.overlap = overlap;
            best = Some((cur, a));
        }
    }

    match best {
        Some((s, a)) if s >= 0.25 => a,
        other => {
            // No convincing lock: assume the same framing, turned to match.
            let portrait_ref = ref_aspect0 > 1.0;
            let portrait_render = render_aspect > 1.0;
            let k = if portrait_ref != portrait_render { 1 } else { 0 };
            let ref_aspect = if k == 1 { 1.0 / ref_aspect0 } else { ref_aspect0 };
            let mut a = pose_alignment(
                Pose { log_scale: 0.0, angle: 0.0, cx: 0.5, cy: 0.5 },
                k,
                ref_aspect,
                render_aspect,
            );
            a.score = other.map(|(s, _)| s).unwrap_or(0.0);
            a.overlap = 1.0;
            a.confident = false;
            a
        }
    }
}

/// The reference warped into the render's framing (outside → dark grey), so
/// it can sit under a render in a before/after slider.
pub fn warp_reference(reference: &RgbImage, a: &Alignment, out_w: u32) -> RgbImage {
    let out_h = (out_w as f32 * a.render_aspect).round() as u32;
    let (rw, rh) = (reference.width() as f32, reference.height() as f32);
    let mut out = RgbImage::from_pixel(out_w, out_h, image::Rgb([24, 24, 26]));
    for y in 0..out_h {
        for x in 0..out_w {
            let (u, v) = a.unmap((x as f32 + 0.5) / out_w as f32, (y as f32 + 0.5) / out_h as f32);
            if !(0.0..1.0).contains(&u) || !(0.0..1.0).contains(&v) {
                continue;
            }
            let (ou, ov) = unrotate(a.rot90, u, v);
            let (px, py) = ((ou * rw) as u32, (ov * rh) as u32);
            out.put_pixel(x, y, *reference.get_pixel(px.min(rw as u32 - 1), py.min(rh as u32 - 1)));
        }
    }
    out
}

// ---------------------------------------------------------------------------
// Comparison
// ---------------------------------------------------------------------------

/// Signed differences, render minus reference (Lab units).
#[derive(Clone, Copy, Debug, Default, Serialize)]
pub struct Stats {
    /// Lightness of the mid-tones.
    pub dl: f32,
    /// Green(−)/magenta(+) and blue(−)/yellow(+) cast, measured on the most
    /// neutral patches.
    pub da: f32,
    pub db: f32,
    /// Lightness of the brightest / darkest quarter, relative to `dl`.
    pub dl_hi: f32,
    pub dl_lo: f32,
    /// Saturation (chroma) of the coloured patches.
    pub dc: f32,
}

#[derive(Clone, Debug, Serialize)]
pub struct Measurement {
    /// What the search minimises: `patch_de + TONE_WEIGHT × tone_err`.
    pub loss: f32,
    /// Robust mean ΔE2000 over the patches (worst 5% ignored).
    pub patch_de: f32,
    /// Mean lightness difference between the two tone curves (L* quantiles
    /// 5%…95% of the same patches): contrast, crushed shadows, highlight
    /// roll-off. ΔE2000 down-weights dark tones and the trim drops the
    /// patches that differ most, so without this a much flatter or much
    /// punchier render could still score well.
    pub tone_err: f32,
    /// Plain weighted mean ΔE2000.
    pub mean_de: f32,
    /// Loss once the global lightness/colour offset is removed — how good
    /// this look could get after WB and exposure are fixed.
    pub shape_loss: f32,
    pub stats: Stats,
}

#[derive(Clone, Copy)]
struct Cell {
    /// Position on the render (normalised).
    x: f32,
    y: f32,
    lab: [f32; 3],
    w: f32,
    /// Patch centre on the (rotated) reference, normalised.
    u: f32,
    v: f32,
}

/// A reference prepared for fast comparison against renders.
pub struct Reference {
    pub alignment: Alignment,
    cells: Vec<Cell>,
    /// Patch blur radius as a fraction of the render width.
    blur_frac: f32,
    /// True when the reference is (nearly) black and white.
    pub monochrome: bool,
    /// Patches dropped because their content isn't in the photo (someone
    /// moved, a different foreground).
    pub dropped: usize,
}

const WORK_W: usize = 400;
/// Weight of the tone-curve error in the loss (see [`Measurement::tone_err`]).
const TONE_WEIGHT: f32 = 0.5;
const GRID_X: usize = 40;

impl Reference {
    pub fn new(reference: &RgbImage, frame: &RgbImage) -> Reference {
        let alignment = align(reference, frame);
        let mut r = Self::with_alignment(reference, alignment);
        r.refine_locally(reference, frame);
        r
    }

    pub fn with_alignment(reference: &RgbImage, alignment: Alignment) -> Reference {
        let k = alignment.rot90;
        let crop = rotate_crop(k, detect_crop(&Img::from_rgb(reference, 256)));
        let img = Img::from_rgb(reference, WORK_W).rotate(k);
        let [u0, v0, u1, v1] = crop;
        let cw = (u1 - u0) / GRID_X as f32;
        let gy = (((v1 - v0) * img.h as f32) / (cw * img.w as f32)).round().max(1.0) as usize;
        let ch = (v1 - v0) / gy as f32;
        let r = ((cw * img.w as f32) / 2.0).round().max(1.0) as usize;
        let blurred = img.blur(r);
        let light = img.lightness();
        let mean_l = Gray { w: img.w, h: img.h, v: box_blur(&light.v, img.w, img.h, r) };
        let sq: Vec<f32> = light.v.iter().map(|x| x * x).collect();
        let mean_sq = box_blur(&sq, img.w, img.h, r);

        let mut cells = Vec::new();
        let mut chroma_sum = 0.0;
        for j in 0..gy {
            for i in 0..GRID_X {
                let u = u0 + (i as f32 + 0.5) * cw;
                let v = v0 + (j as f32 + 0.5) * ch;
                let (x, y) = alignment.map(u, v);
                if !(0.02..0.98).contains(&x) || !(0.02..0.98).contains(&y) {
                    continue;
                }
                let lab = linear_to_lab(blurred.sample(u, v));
                if lab[0] > 98.5 || lab[0] < 1.5 {
                    continue; // clipped in the scan: carries no tone information
                }
                let (px, py) = (
                    ((u * img.w as f32) as usize).min(img.w - 1),
                    ((v * img.h as f32) as usize).min(img.h - 1),
                );
                let idx = py * img.w + px;
                let var = (mean_sq[idx] - mean_l.v[idx].powi(2)).max(0.0);
                // Busy patches (edges, texture) suffer most from misalignment.
                let w = 1.0 / (1.0 + var / 36.0);
                chroma_sum += chroma(&lab) * w;
                cells.push(Cell { x, y, lab, w, u, v });
            }
        }
        let wsum: f32 = cells.iter().map(|c| c.w).sum();
        let monochrome = wsum > 0.0 && chroma_sum / wsum < 3.0;
        Reference {
            blur_frac: alignment.scale * cw / 2.0,
            alignment,
            cells,
            monochrome,
            dropped: 0,
        }
    }

    /// The two pictures are rarely taken from exactly the same spot, at the
    /// same instant, with the same lens: one global transform leaves
    /// parallax and movement. Give every patch its own small offset by block
    /// matching (normalised cross-correlation of lightness, so tone and
    /// colour differences don't matter), fill flat patches (sky, snow) from
    /// their neighbours, and drop textured patches that match nothing.
    pub fn refine_locally(&mut self, reference: &RgbImage, frame: &RgbImage) {
        const R: isize = 12; // window half-size, reference pixels at WORK_W
        const STEP: usize = 2; // window subsampling
        const MIN_NCC: f32 = 0.55;
        let a = self.alignment;
        let ref_l = Img::from_rgb(reference, WORK_W).rotate(a.rot90).lightness();
        let fw = ((WORK_W as f32 / a.scale.max(0.2)).round() as usize).clamp(160, 1400);
        let frame_l = Img::from_rgb(frame, fw).lightness();
        let (fwf, fhf) = (frame_l.w as f32, frame_l.h as f32);
        let max_d = (0.08 * fwf).round() as isize;

        // The window around a patch: frame position (pixels) and reference value.
        let window = |c: &Cell| -> Vec<(f32, f32, f32)> {
            let (cx, cy) = (c.u * ref_l.w as f32, c.v * ref_l.h as f32);
            let mut pts: Vec<(f32, f32, f32)> = Vec::new();
            for wy in (-R..=R).step_by(STEP) {
                for wx in (-R..=R).step_by(STEP) {
                    let (rx, ry) = (cx + wx as f32, cy + wy as f32);
                    if rx < 0.0 || ry < 0.0 || rx >= ref_l.w as f32 || ry >= ref_l.h as f32 {
                        continue;
                    }
                    let val = ref_l.v[ry as usize * ref_l.w + rx as usize];
                    let (x, y) = a.map(rx / ref_l.w as f32, ry / ref_l.h as f32);
                    pts.push((x * fwf - 0.5, y * fhf - 0.5, val));
                }
            }
            pts
        };
        let variance = |v: &mut dyn Iterator<Item = f32>| -> f32 {
            let v: Vec<f32> = v.collect();
            let n = v.len().max(1) as f32;
            let m = v.iter().sum::<f32>() / n;
            v.iter().map(|x| (x - m).powi(2)).sum::<f32>() / n
        };
        // Lightness variance of the photo under a patch's window at an offset.
        let frame_var = |c: &Cell, dx: f32, dy: f32| -> f32 {
            variance(&mut window(c).iter().filter_map(|&(x, y, _)| frame_l.sample_px(x + dx, y + dy)))
        };
        // Per patch: the window's variance on the reference, and the best
        // (ncc, dx, dy) in frame pixels when it has texture to lock onto.
        let search = |c: &Cell| -> (f32, Option<(f32, f32, f32)>) {
            let pts = window(c);
            let n = pts.len() as f32;
            if n < 20.0 {
                return (0.0, None);
            }
            let mean = pts.iter().map(|p| p.2).sum::<f32>() / n;
            let var = pts.iter().map(|p| (p.2 - mean).powi(2)).sum::<f32>() / n;
            if var < 6.0 {
                return (var, None); // flat: nothing to lock onto
            }
            let ncc_at = |dx: f32, dy: f32| -> f32 {
                let (mut sb, mut sbb, mut sab, mut k) = (0f32, 0f32, 0f32, 0f32);
                for &(x, y, av) in &pts {
                    let Some(bv) = frame_l.sample_px(x + dx, y + dy) else { return -1.0 };
                    let av = av - mean;
                    sb += bv;
                    sbb += bv * bv;
                    sab += av * bv;
                    k += 1.0;
                }
                let vb = sbb - sb * sb / k;
                if vb <= 1e-3 { -1.0 } else { sab / (var * n * vb).sqrt() }
            };
            let mut best = (-1.0f32, 0.0f32, 0.0f32);
            for dy in (-max_d..=max_d).step_by(2) {
                for dx in (-max_d..=max_d).step_by(2) {
                    let s = ncc_at(dx as f32, dy as f32);
                    if s > best.0 {
                        best = (s, dx as f32, dy as f32);
                    }
                }
            }
            let (_, bx, by) = best;
            for dy in [-1.0f32, 0.0, 1.0] {
                for dx in [-1.0f32, 0.0, 1.0] {
                    let s = ncc_at(bx + dx, by + dy);
                    if s > best.0 {
                        best = (s, bx + dx, by + dy);
                    }
                }
            }
            (var, Some(best))
        };

        let threads = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(4);
        let chunk = self.cells.len().div_ceil(threads).max(1);
        let found: Vec<(f32, Option<(f32, f32, f32)>)> = std::thread::scope(|sc| {
            let handles: Vec<_> = self.cells.chunks(chunk).map(|cs| sc.spawn(move || cs.iter().map(search).collect::<Vec<_>>())).collect();
            handles.into_iter().flat_map(|h| h.join().unwrap()).collect()
        });

        let good: Vec<(usize, f32, f32)> = found
            .iter()
            .enumerate()
            .filter_map(|(i, f)| f.1.filter(|b| b.0 >= MIN_NCC).map(|b| (i, b.1, b.2)))
            .collect();
        let median = |mut v: Vec<f32>| -> f32 {
            if v.is_empty() {
                return 0.0;
            }
            v.sort_by(f32::total_cmp);
            v[v.len() / 2]
        };
        // Offset field at a patch: median of the locked patches nearby.
        let near = |i: usize, radius: f32| -> Option<(f32, f32)> {
            let c = &self.cells[i];
            let (xs, ys): (Vec<f32>, Vec<f32>) = good
                .iter()
                .filter(|(j, _, _)| {
                    let o = &self.cells[*j];
                    ((o.u - c.u) * 40.0).hypot((o.v - c.v) * 40.0 * a.ref_aspect) <= radius
                })
                .map(|&(_, dx, dy)| (dx, dy))
                .unzip();
            (xs.len() >= 3).then(|| (median(xs), median(ys)))
        };

        let mut keep = Vec::with_capacity(self.cells.len());
        let mut dropped = 0;
        for (i, (var, m)) in found.iter().enumerate() {
            let offset = match m {
                // Textured and locked: trust it unless it disagrees with its
                // neighbourhood (a repeated pattern matched the wrong copy).
                Some((s, dx, dy)) if *s >= MIN_NCC => match near(i, 4.0) {
                    Some((mx, my)) if (dx - mx).hypot(dy - my) > 0.03 * fwf => None,
                    _ => Some((*dx, *dy)),
                },
                // Strongly textured but nothing like it in the photo.
                _ if *var > 40.0 => None,
                // Flat, or just grain and gentle gradients: follow locked
                // neighbours, and only if the photo is flat there too (not a
                // shadow edge or someone standing in the other shot).
                _ => near(i, 6.0).filter(|&(dx, dy)| frame_var(&self.cells[i], dx, dy) <= (4.0 * var).max(40.0)),
            };
            match offset {
                Some((dx, dy)) => {
                    let c = &self.cells[i];
                    keep.push(Cell { x: c.x + dx / fwf, y: c.y + dy / fhf, ..*c });
                }
                None => dropped += 1,
            }
        }
        // Too little left means the lock-on failed; keep the global mapping.
        if keep.len() >= self.cells.len() / 3 && keep.len() >= 60 {
            keep.retain(|c| (0.0..1.0).contains(&c.x) && (0.0..1.0).contains(&c.y));
            self.cells = keep;
            self.dropped = dropped;
        }
    }

    /// Per-patch errors for a render: (x, y on the render, ΔE, weight).
    pub fn patch_errors(&self, render: &RgbImage) -> Vec<(f32, f32, f32, f32)> {
        let img = Img::from_rgb(render, WORK_W);
        let r = (self.blur_frac * img.w as f32).round().max(1.0) as usize;
        let blurred = img.blur(r);
        self.cells
            .iter()
            .map(|c| (c.x, c.y, delta_e2000(linear_to_lab(blurred.sample(c.x, c.y)), c.lab), c.w))
            .collect()
    }

    pub fn cell_count(&self) -> usize {
        self.cells.len()
    }

    pub fn measure(&self, render: &RgbImage) -> Measurement {
        let img = Img::from_rgb(render, WORK_W);
        let r = (self.blur_frac * img.w as f32).round().max(1.0) as usize;
        let blurred = img.blur(r);
        let labs: Vec<[f32; 3]> = self
            .cells
            .iter()
            .map(|c| linear_to_lab(blurred.sample(c.x, c.y)))
            .collect();
        self.score(&labs)
    }

    fn score(&self, labs: &[[f32; 3]]) -> Measurement {
        let cells = &self.cells;
        let wsum: f32 = cells.iter().map(|c| c.w).sum::<f32>().max(1e-6);
        // Robust signed differences: weighted mean of the middle 60%, so
        // patches where the scene itself changed (a shadow that moved,
        // someone walking by) don't drag the knobs around.
        let wmean = |f: &dyn Fn(&Cell, &[f32; 3]) -> Option<f32>| -> Option<f32> {
            let mut v: Vec<(f32, f32)> =
                cells.iter().zip(labs).filter_map(|(c, l)| f(c, l).map(|x| (x, c.w))).collect();
            let total: f32 = v.iter().map(|x| x.1).sum();
            if total <= 0.0 {
                return None;
            }
            v.sort_by(|a, b| a.0.total_cmp(&b.0));
            let (lo, hi) = (0.2 * total, 0.8 * total);
            let (mut acc, mut s, mut n) = (0.0f32, 0.0f32, 0.0f32);
            for (x, w) in v {
                let take = (acc + w).min(hi) - acc.max(lo);
                if take > 0.0 {
                    s += x * take;
                    n += take;
                }
                acc += w;
            }
            (n > 0.0).then(|| s / n)
        };
        let dl_all = wmean(&|c, l| Some(l[0] - c.lab[0])).unwrap_or(0.0); // fallback only
        let da_all = wmean(&|c, l| Some(l[1] - c.lab[1])).unwrap_or(0.0);
        let db_all = wmean(&|c, l| Some(l[2] - c.lab[2])).unwrap_or(0.0);

        // WB is judged on neutral patches when there are enough of them.
        let neutral_w: f32 = cells.iter().filter(|c| chroma(&c.lab) < 12.0).map(|c| c.w).sum();
        let (da, db) = if neutral_w > 0.15 * wsum {
            (
                wmean(&|c, l| (chroma(&c.lab) < 12.0).then(|| l[1] - c.lab[1])).unwrap_or(0.0),
                wmean(&|c, l| (chroma(&c.lab) < 12.0).then(|| l[2] - c.lab[2])).unwrap_or(0.0),
            )
        } else {
            (da_all, db_all)
        };

        let mut ls: Vec<f32> = cells.iter().map(|c| c.lab[0]).collect();
        ls.sort_by(f32::total_cmp);
        let q = |p: f32| ls.get(((ls.len() as f32 - 1.0) * p) as usize).copied().unwrap_or(50.0);
        let (q25, q75) = (q(0.25), q(0.75));
        // Exposure is judged on the mid-tones; the ends belong to Highlight/Shadow.
        let dl = wmean(&|c, l| (c.lab[0] > q25 && c.lab[0] < q75).then(|| l[0] - c.lab[0])).unwrap_or(dl_all);
        let dl_hi = wmean(&|c, l| (c.lab[0] >= q75).then(|| l[0] - c.lab[0])).unwrap_or(dl) - dl;
        let dl_lo = wmean(&|c, l| (c.lab[0] <= q25).then(|| l[0] - c.lab[0])).unwrap_or(dl) - dl;
        let dc = wmean(&|c, l| (chroma(&c.lab) > 8.0).then(|| chroma(l) - chroma(&c.lab)))
            .or_else(|| wmean(&|c, l| Some(chroma(l) - chroma(&c.lab))))
            .unwrap_or(0.0);

        let des: Vec<(f32, f32)> = cells.iter().zip(labs).map(|(c, l)| (delta_e2000(*l, c.lab), c.w)).collect();
        let mean_de = des.iter().map(|(d, w)| d * w).sum::<f32>() / wsum;
        let patch_de = trimmed_mean(des, 0.05);
        let mut lr: Vec<f32> = labs.iter().map(|l| l[0]).collect();
        lr.sort_by(f32::total_cmp);
        let qr = |p: f32| lr.get(((lr.len() as f32 - 1.0) * p) as usize).copied().unwrap_or(50.0);
        let tone_err = (1..=19).map(|i| (qr(i as f32 * 0.05) - q(i as f32 * 0.05)).abs()).sum::<f32>() / 19.0;
        let loss = patch_de + TONE_WEIGHT * tone_err;
        let shape_loss = self.shape_loss(labs, da_all, db_all);

        Measurement {
            loss,
            patch_de,
            tone_err,
            mean_de,
            shape_loss,
            stats: Stats { dl, da, db, dl_hi, dl_lo, dc },
        }
    }
}

impl Reference {
    /// The error left once everything WB, exposure, tone and Color can fix is
    /// taken out: lightness is mapped by rank onto the reference's (any
    /// monotonic tone curve), the mean cast is removed and saturation is
    /// scaled to match. What remains is the look's own hue rendering.
    fn shape_loss(&self, labs: &[[f32; 3]], da: f32, db: f32) -> f32 {
        let n = labs.len();
        if n == 0 {
            return 0.0;
        }
        let mut order: Vec<usize> = (0..n).collect();
        order.sort_by(|a, b| labs[*a][0].total_cmp(&labs[*b][0]));
        let mut ref_l: Vec<f32> = self.cells.iter().map(|c| c.lab[0]).collect();
        ref_l.sort_by(f32::total_cmp);
        let mut mapped_l = vec![0f32; n];
        for (rank, &i) in order.iter().enumerate() {
            mapped_l[i] = ref_l[rank];
        }
        let wsum: f32 = self.cells.iter().map(|c| c.w).sum::<f32>().max(1e-6);
        let c_ref: f32 = self.cells.iter().map(|c| chroma(&c.lab) * c.w).sum::<f32>() / wsum;
        let c_ren: f32 = self
            .cells
            .iter()
            .zip(labs)
            .map(|(c, l)| (l[1] - da).hypot(l[2] - db) * c.w)
            .sum::<f32>()
            / wsum;
        let k = if c_ren > 0.5 { (c_ref / c_ren).clamp(0.7, 1.4) } else { 1.0 };
        let des = self
            .cells
            .iter()
            .zip(labs)
            .zip(mapped_l)
            .map(|((c, l), ml)| (delta_e2000([ml, (l[1] - da) * k, (l[2] - db) * k], c.lab), c.w))
            .collect();
        trimmed_mean(des, 0.10)
    }
}

/// Weighted mean ignoring the worst `trim` share of the weight.
fn trimmed_mean(mut v: Vec<(f32, f32)>, trim: f32) -> f32 {
    v.sort_by(|a, b| a.0.total_cmp(&b.0));
    let total: f32 = v.iter().map(|x| x.1).sum();
    let keep = total * (1.0 - trim);
    let (mut s, mut n) = (0.0, 0.0);
    for (d, w) in v {
        if n >= keep {
            break;
        }
        let take = w.min(keep - n);
        s += d * take;
        n += take;
    }
    if n > 0.0 { s / n } else { 0.0 }
}

// ---------------------------------------------------------------------------
// Parameters
// ---------------------------------------------------------------------------

#[derive(Clone, Debug, PartialEq)]
pub enum Wb {
    Auto,
    Daylight,
    Kelvin(u32),
    /// Any other camera preset (shade, incandescent…) kept from a start recipe.
    Preset(String),
}

/// Settings that don't change colour or tone at the patch scale; carried as is.
#[derive(Clone, Debug, PartialEq)]
pub struct Texture {
    pub grain: GrainEffect,
    pub grain_size: GrainSize,
    pub noise_reduction: i32,
    pub sharpness: i32,
    pub clarity: i32,
}

impl Default for Texture {
    /// A film-like default: some fine grain, soft rendering.
    fn default() -> Self {
        Texture {
            grain: GrainEffect::Weak,
            grain_size: GrainSize::Small,
            noise_reduction: -4,
            sharpness: -1,
            clarity: 0,
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct Params {
    pub film_sim: FilmSimulation,
    pub wb: Wb,
    pub wb_r: i32,
    pub wb_b: i32,
    /// Exposure compensation in thirds of a stop.
    pub ev_thirds: i32,
    pub highlight: f32,
    pub shadow: f32,
    pub color: i32,
    pub chrome: ChromeLevel,
    pub blue: ChromeLevel,
    pub dr: u32,
    pub texture: Texture,
}

pub const COLOR_SIMS: [FilmSimulation; 11] = [
    FilmSimulation::ClassicNeg,
    FilmSimulation::NostalgicNeg,
    FilmSimulation::ClassicChrome,
    FilmSimulation::RealaAce,
    FilmSimulation::ProNegStd,
    FilmSimulation::ProNegHi,
    FilmSimulation::Provia,
    FilmSimulation::Astia,
    FilmSimulation::Velvia,
    FilmSimulation::Eterna,
    FilmSimulation::EternaBleachBypass,
];

pub const MONO_SIMS: [FilmSimulation; 8] = [
    FilmSimulation::Acros,
    FilmSimulation::AcrosYe,
    FilmSimulation::AcrosR,
    FilmSimulation::AcrosG,
    FilmSimulation::Monochrome,
    FilmSimulation::MonochromeYe,
    FilmSimulation::MonochromeR,
    FilmSimulation::MonochromeG,
];

pub fn sim_slug(s: FilmSimulation) -> &'static str {
    use FilmSimulation::*;
    match s {
        Provia => "provia",
        Velvia => "velvia",
        Astia => "astia",
        ClassicChrome => "classic-chrome",
        ClassicNeg => "classic-neg",
        ProNegHi => "pro-neg-hi",
        ProNegStd => "pro-neg-std",
        Eterna => "eterna",
        EternaBleachBypass => "eterna-bleach-bypass",
        Acros => "acros",
        AcrosYe => "acros-ye",
        AcrosR => "acros-r",
        AcrosG => "acros-g",
        Monochrome => "monochrome",
        MonochromeYe => "monochrome-ye",
        MonochromeR => "monochrome-r",
        MonochromeG => "monochrome-g",
        Sepia => "sepia",
        NostalgicNeg => "nostalgic-neg",
        RealaAce => "reala-ace",
    }
}

fn level_str(l: ChromeLevel) -> &'static str {
    match l {
        ChromeLevel::Off => "off",
        ChromeLevel::Weak => "weak",
        ChromeLevel::Strong => "strong",
    }
}

fn thirds_to_millis(t: i32) -> i32 {
    let a = t.abs();
    t.signum() * (a / 3 * 1000 + [0, 333, 667][(a % 3) as usize])
}

fn thirds_label(t: i32) -> String {
    let sign = if t < 0 { "-" } else { "+" };
    let (w, r) = (t.abs() / 3, t.abs() % 3);
    match (w, r) {
        (0, 0) => "0".into(),
        (0, r) => format!("{sign}{r}/3"),
        (w, 0) => format!("{sign}{w}"),
        (w, r) => format!("{sign}{w} {r}/3"),
    }
}

impl Params {
    pub fn neutral(film_sim: FilmSimulation, dr: u32) -> Params {
        Params {
            film_sim,
            wb: Wb::Auto,
            wb_r: 0,
            wb_b: 0,
            ev_thirds: 0,
            highlight: 0.0,
            shadow: 0.0,
            color: 0,
            chrome: ChromeLevel::Off,
            blue: ChromeLevel::Off,
            dr,
            texture: Texture::default(),
        }
    }

    /// Start from an existing recipe (its exposure advice is ignored).
    pub fn from_settings(s: &RecipeSettings, dr: u32) -> Params {
        let wb = match (s.white_balance.as_deref(), s.wb_temp) {
            (Some("temperature"), Some(k)) => Wb::Kelvin(k),
            (Some("daylight"), _) => Wb::Daylight,
            (None | Some("auto"), _) => Wb::Auto,
            (Some(other), _) => Wb::Preset(other.to_string()),
        };
        Params {
            film_sim: s.film_sim.unwrap_or(FilmSimulation::ClassicNeg),
            wb,
            wb_r: s.wb_shift_r.unwrap_or(0),
            wb_b: s.wb_shift_b.unwrap_or(0),
            ev_thirds: 0,
            highlight: s.highlight.unwrap_or(0.0),
            shadow: s.shadow.unwrap_or(0.0),
            color: s.color.unwrap_or(0),
            chrome: s.chrome_effect.unwrap_or(ChromeLevel::Off),
            blue: s.chrome_blue.unwrap_or(ChromeLevel::Off),
            dr: s.dynamic_range.unwrap_or(dr),
            texture: Texture {
                grain: s.grain.unwrap_or(GrainEffect::Off),
                grain_size: s.grain_size.unwrap_or(GrainSize::Small),
                noise_reduction: s.noise_reduction.unwrap_or(0),
                sharpness: s.sharpness.unwrap_or(0),
                clarity: s.clarity.unwrap_or(0),
            },
        }
    }

    /// Every field set explicitly, so nothing falls back to the as-shot value.
    pub fn to_settings(&self) -> RecipeSettings {
        let (white_balance, wb_temp) = match &self.wb {
            Wb::Auto => ("auto".to_string(), None),
            Wb::Daylight => ("daylight".to_string(), None),
            Wb::Kelvin(k) => ("temperature".to_string(), Some(*k)),
            Wb::Preset(p) => (p.clone(), None),
        };
        RecipeSettings {
            film_sim: Some(self.film_sim),
            grain: Some(self.texture.grain),
            grain_size: Some(self.texture.grain_size),
            highlight: Some(self.highlight),
            shadow: Some(self.shadow),
            color: Some(self.color),
            sharpness: Some(self.texture.sharpness),
            noise_reduction: Some(self.texture.noise_reduction),
            clarity: Some(self.texture.clarity),
            white_balance: Some(white_balance),
            wb_temp,
            wb_shift_r: Some(self.wb_r),
            wb_shift_b: Some(self.wb_b),
            chrome_effect: Some(self.chrome),
            chrome_blue: Some(self.blue),
            dynamic_range: Some(self.dr),
            exposure_comp: Some(thirds_to_millis(self.ev_thirds)),
        }
    }

    fn key(&self) -> String {
        format!("{self:?}")
    }

    fn wb_label(&self) -> String {
        match &self.wb {
            Wb::Auto => "Auto".into(),
            Wb::Daylight => "Daylight".into(),
            Wb::Kelvin(k) => format!("{k}K"),
            Wb::Preset(p) => p.clone(),
        }
    }

    /// Short description of what differs from `other` ("Color +1 → +3").
    pub fn diff_label(&self, other: &Params) -> String {
        let mut d = Vec::new();
        let pm = |v: f32| if v == 0.0 { "0".to_string() } else { format!("{v:+}") };
        if self.film_sim != other.film_sim {
            d.push(self.film_sim.label().to_string());
        }
        if self.wb != other.wb {
            d.push(format!("WB {}", self.wb_label()));
        }
        if self.wb_r != other.wb_r || self.wb_b != other.wb_b {
            d.push(format!("R{:+} B{:+}", self.wb_r, self.wb_b));
        }
        if self.ev_thirds != other.ev_thirds {
            d.push(format!("EV {}", thirds_label(self.ev_thirds)));
        }
        if self.highlight != other.highlight {
            d.push(format!("Highlight {}", pm(self.highlight)));
        }
        if self.shadow != other.shadow {
            d.push(format!("Shadow {}", pm(self.shadow)));
        }
        if self.color != other.color {
            d.push(format!("Color {}", pm(self.color as f32)));
        }
        if self.chrome != other.chrome {
            d.push(format!("Chrome {}", level_str(self.chrome)));
        }
        if self.blue != other.blue {
            d.push(format!("FX Blue {}", level_str(self.blue)));
        }
        if self.dr != other.dr {
            d.push(format!("DR{}", self.dr));
        }
        if d.is_empty() { "same".into() } else { d.join(", ") }
    }

    /// The settings in the recipes.json schema.
    pub fn to_recipe_json(&self, name: &str, slug: &str) -> Value {
        let (white_balance, wb_temp) = match &self.wb {
            Wb::Auto => ("auto".to_string(), None),
            Wb::Daylight => ("daylight".to_string(), None),
            Wb::Kelvin(k) => ("temperature".to_string(), Some(*k)),
            Wb::Preset(p) => (p.clone(), None),
        };
        let mut v = json!({
            "name": name,
            "slug": slug,
            "film_sim": sim_slug(self.film_sim),
            "highlight": self.highlight,
            "shadow": self.shadow,
            "color": self.color,
            "noise_reduction": self.texture.noise_reduction,
            "sharpness": self.texture.sharpness,
            "clarity": self.texture.clarity,
            "chrome_effect": level_str(self.chrome),
            "chrome_blue": level_str(self.blue),
            "dynamic_range": self.dr,
            "white_balance": white_balance,
            "wb_shift_r": self.wb_r,
            "wb_shift_b": self.wb_b,
            "exposure_comp": thirds_label(self.ev_thirds),
        });
        if self.texture.grain != GrainEffect::Off {
            v["grain"] = json!(match self.texture.grain {
                GrainEffect::Strong => "strong",
                _ => "weak",
            });
            v["grain_size"] = json!(match self.texture.grain_size {
                GrainSize::Large => "large",
                GrainSize::Small => "small",
            });
        }
        if let Some(k) = wb_temp {
            v["wb_temp"] = json!(k);
        }
        v
    }
}

// ---------------------------------------------------------------------------
// Search
// ---------------------------------------------------------------------------

pub struct Rendered {
    pub image: RgbImage,
    pub jpeg: Vec<u8>,
}

#[derive(Clone, Debug)]
pub struct Eval {
    /// 1-based render number.
    pub n: usize,
    pub phase: String,
    pub label: String,
    pub params: Params,
    pub m: Measurement,
}

pub enum Event<'a> {
    Phase(&'a str),
    Evaluated { eval: &'a Eval, rendered: &'a Rendered, is_best: bool },
}

pub struct SearchOptions {
    pub max_renders: usize,
    /// Start from these settings instead of sweeping the film simulations.
    pub start: Option<Params>,
    /// DR values this RAF can be developed with.
    pub dr_options: Vec<u32>,
    /// DR to use before DR itself is tuned (normally the as-shot one).
    pub default_dr: u32,
    /// Stop once the error is this small.
    pub good_enough: f32,
}

impl Default for SearchOptions {
    fn default() -> Self {
        SearchOptions {
            max_renders: 60,
            start: None,
            dr_options: vec![100],
            default_dr: 100,
            good_enough: 1.0,
        }
    }
}

pub struct SearchResult {
    pub best: Eval,
    pub renders: usize,
    pub stopped: bool,
    pub budget_exhausted: bool,
    /// A camera error that ended the search early (the best so far is kept).
    pub error: Option<String>,
}

enum Halt {
    Stopped,
    Budget,
    Done,
    Error(String),
}

type RenderFn<'r> = dyn FnMut(&RecipeSettings) -> Result<Rendered, String> + 'r;
type EventFn<'r> = dyn FnMut(Event) + 'r;

struct Search<'a, 'r> {
    reference: &'a Reference,
    opts: &'a SearchOptions,
    stop: &'a AtomicBool,
    render: &'a mut RenderFn<'r>,
    on_event: &'a mut EventFn<'r>,
    evals: Vec<Eval>,
    seen: HashMap<String, usize>,
    best: Option<usize>,
    phase: String,
    renders: usize,
    /// d(a,b cast)/d(R,B shift), measured once and reused.
    wb_jacobian: Option<[[f32; 2]; 2]>,
    /// d(stat)/d(knob) per knob, measured once and reused.
    slopes: HashMap<Knob, f32>,
    /// Polish moves (knob, value×100, direction) that clearly made things
    /// worse; not retried while the knob is still at that value.
    failed: HashMap<(Knob, i32, i8), f32>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
enum Knob {
    WbR,
    WbB,
    Kelvin,
    Exposure,
    Highlight,
    Shadow,
    Color,
}

impl Knob {
    fn get(self, p: &Params) -> f32 {
        match self {
            Knob::WbR => p.wb_r as f32,
            Knob::WbB => p.wb_b as f32,
            Knob::Kelvin => match p.wb {
                Wb::Kelvin(k) => k as f32,
                _ => 5500.0,
            },
            Knob::Exposure => p.ev_thirds as f32,
            Knob::Highlight => p.highlight,
            Knob::Shadow => p.shadow,
            Knob::Color => p.color as f32,
        }
    }

    fn set(self, p: &Params, v: f32) -> Params {
        let mut q = p.clone();
        match self {
            Knob::WbR => q.wb_r = v.round() as i32,
            Knob::WbB => q.wb_b = v.round() as i32,
            Knob::Kelvin => q.wb = Wb::Kelvin(v.round() as u32),
            Knob::Exposure => q.ev_thirds = v.round() as i32,
            Knob::Highlight => q.highlight = v,
            Knob::Shadow => q.shadow = v,
            Knob::Color => q.color = v.round() as i32,
        }
        q
    }

    /// (min, max, step)
    fn range(self) -> (f32, f32, f32) {
        match self {
            Knob::WbR | Knob::WbB => (-9.0, 9.0, 1.0),
            Knob::Kelvin => (2500.0, 10000.0, 100.0),
            Knob::Exposure => (-6.0, 6.0, 1.0),
            Knob::Highlight | Knob::Shadow => (-2.0, 4.0, 0.5),
            Knob::Color => (-4.0, 4.0, 1.0),
        }
    }

    /// Size of the calibration probe, in steps.
    fn probe(self) -> f32 {
        match self {
            Knob::Kelvin => 6.0,
            Knob::Exposure => 1.0,
            _ => 2.0,
        }
    }

    /// The measured difference this knob should cancel, and the expected
    /// sign of d(stat)/d(knob) (only used to pick the first probe's side).
    fn stat(self, s: &Stats) -> (f32, f32) {
        match self {
            Knob::WbR => (s.da, 1.0),
            Knob::WbB => (s.db, -1.0),
            Knob::Kelvin => (s.db, 1.0),
            Knob::Exposure => (s.dl, 1.0),
            Knob::Highlight => (s.dl_hi, 1.0),
            Knob::Shadow => (s.dl_lo, -1.0),
            Knob::Color => (s.dc, 1.0),
        }
    }

    /// Differences smaller than this aren't worth a render.
    fn tolerance(self) -> f32 {
        match self {
            Knob::WbR | Knob::WbB | Knob::Kelvin => 0.5,
            _ => 0.7,
        }
    }

    fn snap(self, v: f32) -> f32 {
        let (lo, hi, step) = self.range();
        ((v / step).round() * step).clamp(lo, hi)
    }
}

impl Search<'_, '_> {
    fn loss(&self, i: usize) -> f32 {
        self.evals[i].m.loss
    }

    fn set_phase(&mut self, phase: &str) {
        self.phase = phase.to_string();
        (self.on_event)(Event::Phase(phase));
    }

    /// Render and measure `p` (cached). Returns the index into `evals`.
    fn eval(&mut self, p: Params, label: &str) -> Result<usize, Halt> {
        if let Some(&i) = self.seen.get(&p.key()) {
            return Ok(i);
        }
        if self.stop.load(Ordering::Relaxed) {
            return Err(Halt::Stopped);
        }
        if self.renders >= self.opts.max_renders {
            return Err(Halt::Budget);
        }
        let rendered = (self.render)(&p.to_settings()).map_err(Halt::Error)?;
        self.renders += 1;
        let m = self.reference.measure(&rendered.image);
        let idx = self.evals.len();
        self.seen.insert(p.key(), idx);
        self.evals.push(Eval {
            n: self.renders,
            phase: self.phase.clone(),
            label: label.to_string(),
            params: p,
            m,
        });
        let is_best = self.best.is_none_or(|b| self.loss(idx) < self.loss(b));
        if is_best {
            self.best = Some(idx);
        }
        (self.on_event)(Event::Evaluated { eval: &self.evals[idx], rendered: &rendered, is_best });
        if is_best && self.loss(idx) <= self.opts.good_enough {
            return Err(Halt::Done);
        }
        Ok(idx)
    }

    /// Evaluate `p`, labelled by what changed relative to eval `from`.
    fn eval_from(&mut self, from: usize, p: Params) -> Result<usize, Halt> {
        let label = p.diff_label(&self.evals[from].params);
        self.eval(p, &label)
    }

    /// Tune one numeric knob starting from eval `from`: jump to the value
    /// where the measured difference should vanish (the knob's sensitivity
    /// is measured with one probe step the first time), then keep stepping
    /// while it helps. Returns the best eval of the values tried.
    fn tune(&mut self, knob: Knob, from: usize) -> Result<usize, Halt> {
        let base = self.evals[from].params.clone();
        let (lo, hi, step) = knob.range();
        let v0 = knob.get(&base);
        let (s0, sign) = knob.stat(&self.evals[from].m.stats);
        if s0.abs() < knob.tolerance() {
            return Ok(from);
        }
        let mut tried: Vec<(f32, usize)> = vec![(v0, from)];
        let target = match self.slopes.get(&knob) {
            Some(&k) => knob.snap(v0 - s0 / k),
            None => {
                let v1 = knob.snap(v0 - s0.signum() * sign * step * knob.probe());
                if v1 == v0 {
                    return Ok(from);
                }
                let i1 = self.eval_from(from, knob.set(&base, v1))?;
                tried.push((v1, i1));
                let (s1, _) = knob.stat(&self.evals[i1].m.stats);
                let slope = (s1 - s0) / (v1 - v0);
                if slope * sign > 1e-3 {
                    self.slopes.insert(knob, slope);
                    knob.snap(v0 - s0 / slope)
                } else {
                    v1 // the knob doesn't act as expected here: don't extrapolate
                }
            }
        };
        if tried.iter().all(|(v, _)| (*v - target).abs() > 1e-6) {
            let i = self.eval_from(from, knob.set(&base, target))?;
            tried.push((target, i));
            let (st, _) = knob.stat(&self.evals[i].m.stats);
            let slope = (st - s0) / (target - v0);
            if slope * sign > 1e-3 {
                self.slopes.insert(knob, slope);
            }
        }
        // Keep stepping from the best value while the measurement still
        // points somewhere and it keeps improving.
        for _ in 0..2 {
            let &(vb, ib) = tried.iter().min_by(|a, b| self.loss(a.1).total_cmp(&self.loss(b.1))).unwrap();
            let (sb, sign) = knob.stat(&self.evals[ib].m.stats);
            if sb.abs() < knob.tolerance() {
                break;
            }
            let v = vb - sb.signum() * sign * step;
            if v < lo - 1e-6 || v > hi + 1e-6 || tried.iter().any(|(t, _)| (*t - v).abs() < 1e-6) {
                break;
            }
            let i = self.eval_from(from, knob.set(&base, v))?;
            tried.push((v, i));
            if self.loss(i) >= self.loss(ib) {
                break;
            }
        }
        Ok(tried.iter().map(|t| t.1).min_by(|a, b| self.loss(*a).total_cmp(&self.loss(*b))).unwrap())
    }

    /// Loss-driven polish: one step either way on every knob, following any
    /// direction that helps. Returns whether anything improved.
    fn polish(&mut self, knobs: &[Knob], both_ways: bool) -> Result<bool, Halt> {
        let mut improved = false;
        for &knob in knobs {
            let from = self.best.unwrap();
            let base = self.evals[from].params.clone();
            if knob == Knob::Kelvin && !matches!(base.wb, Wb::Kelvin(_)) {
                continue;
            }
            let (lo, hi, step) = knob.range();
            let v0 = knob.get(&base);
            let (s, sign) = knob.stat(&self.evals[from].m.stats);
            let first = if s.abs() >= knob.tolerance() * 0.5 { -s.signum() * sign } else { 1.0 };
            let dirs: &[f32] = if both_ways { &[first, -first] } else { &[first] };
            for &d in dirs {
                let mut v = v0 + d * step;
                let move_key = (knob, (v0 * 100.0).round() as i32, d as i8);
                if v < lo - 1e-6 || v > hi + 1e-6 || self.failed.get(&move_key).is_some_and(|w| *w > 0.5) {
                    continue;
                }
                let mut cur = from;
                loop {
                    let i = self.eval_from(from, knob.set(&base, v))?;
                    if self.loss(i) >= self.loss(cur) {
                        if cur == from {
                            self.failed.insert(move_key, self.loss(i) - self.loss(from));
                        }
                        break;
                    }
                    cur = i;
                    v += d * step;
                    if v < lo - 1e-6 || v > hi + 1e-6 {
                        break;
                    }
                }
                if cur != from {
                    improved = true;
                    break;
                }
            }
        }
        Ok(improved)
    }

    /// The R/B shifts that should cancel the cast measured on eval `i`,
    /// starting from the shifts in `base`.
    fn wb_correction(&self, i: usize, base: &Params) -> Option<(i32, i32)> {
        let s = self.evals[i].m.stats;
        let (r, b) = solve2(self.wb_jacobian?, [-s.da, -s.db])?;
        Some((
            (base.wb_r as f32 + r).round().clamp(-9.0, 9.0) as i32,
            (base.wb_b as f32 + b).round().clamp(-9.0, 9.0) as i32,
        ))
    }

    /// White balance from eval `from`: probe R and B once to learn how they
    /// move the (a, b) cast, then solve for the shifts that cancel it.
    fn tune_wb(&mut self, from: usize) -> Result<usize, Halt> {
        let base = self.evals[from].params.clone();
        let s0 = self.evals[from].m.stats;
        if s0.da.abs() < 0.5 && s0.db.abs() < 0.5 {
            return Ok(from);
        }
        let mut cands = vec![from];
        if self.wb_jacobian.is_none() {
            let pr = Params { wb_r: (base.wb_r + if s0.da > 0.0 { -2 } else { 2 }).clamp(-9, 9), ..base.clone() };
            let pb = Params { wb_b: (base.wb_b + if s0.db > 0.0 { 2 } else { -2 }).clamp(-9, 9), ..base.clone() };
            let (dr, db) = ((pr.wb_r - base.wb_r) as f32, (pb.wb_b - base.wb_b) as f32);
            let ir = self.eval_from(from, pr)?;
            let ib = self.eval_from(from, pb)?;
            cands.extend([ir, ib]);
            let (sr, sb) = (self.evals[ir].m.stats, self.evals[ib].m.stats);
            if dr != 0.0 && db != 0.0 {
                self.wb_jacobian =
                    Some([[(sr.da - s0.da) / dr, (sb.da - s0.da) / db], [(sr.db - s0.db) / dr, (sb.db - s0.db) / db]]);
            }
        }
        if let Some((r, b)) = self.wb_correction(from, &base) {
            cands.push(self.eval_from(from, Params { wb_r: r, wb_b: b, ..base })?);
        }
        Ok(cands.into_iter().min_by(|a, b| self.loss(*a).total_cmp(&self.loss(*b))).unwrap())
    }

    /// Try each value of a categorical setting on the current best.
    fn try_values<T: Clone>(&mut self, values: &[T], apply: impl Fn(&Params, T) -> Params) -> Result<(), Halt> {
        let from = self.best.unwrap();
        let base = self.evals[from].params.clone();
        for v in values {
            let p = apply(&base, v.clone());
            if p != base {
                self.eval_from(from, p)?;
            }
        }
        Ok(())
    }

    fn run(&mut self) -> Result<(), Halt> {
        let mono = self.reference.monochrome;
        match self.opts.start.clone() {
            Some(p) => {
                self.set_phase("Start recipe");
                self.eval(p, "start recipe")?;
            }
            None => {
                self.set_phase("Film simulations");
                let sims: &[FilmSimulation] = if mono { &MONO_SIMS } else { &COLOR_SIMS };
                let mut idx = Vec::new();
                for &sim in sims {
                    idx.push(self.eval(Params::neutral(sim, self.opts.default_dr), sim.label())?);
                }
                // Rank by how close each look could get once WB and exposure
                // are fixed, not by the raw error.
                idx.sort_by(|a, b| self.evals[*a].m.shape_loss.total_cmp(&self.evals[*b].m.shape_loss));
                if !mono {
                    self.set_phase("White balance");
                    // Calibrate WB and exposure on the most promising look…
                    let w = self.tune_wb(idx[0])?;
                    let w = self.tune(Knob::Exposure, w)?;
                    let fix = self.evals[w].params.clone();
                    // …and give the runners-up the same kind of correction.
                    for &i in &idx[1..3] {
                        let base = self.evals[i].params.clone();
                        let (r, b) = self.wb_correction(i, &base).unwrap_or((fix.wb_r, fix.wb_b));
                        let p = Params { wb_r: r, wb_b: b, ev_thirds: fix.ev_thirds, ..base };
                        let label = format!("{} + WB/EV", p.film_sim.label());
                        self.eval(p, &label)?;
                    }
                }
            }
        }

        if !mono {
            // WB mode: daylight (what colour film is balanced for) vs auto,
            // each with its own cast corrected.
            self.set_phase("White balance");
            let from = self.best.unwrap();
            let base = self.evals[from].params.clone();
            if matches!(base.wb, Wb::Auto | Wb::Daylight) {
                let other = if base.wb == Wb::Auto { Wb::Daylight } else { Wb::Auto };
                let i = self.eval_from(from, Params { wb: other, ..base.clone() })?;
                if let Some((r, b)) = self.wb_correction(i, &base) {
                    let p = Params { wb_r: r, wb_b: b, ..self.evals[i].params.clone() };
                    self.eval_from(from, p)?;
                }
            }
            self.tune_wb(self.best.unwrap())?;
            // Shifts pinned at the limit: the cast is too strong for them —
            // switch to an explicit colour temperature.
            let from = self.best.unwrap();
            let b = self.evals[from].params.clone();
            if b.wb_r.abs() == 9 || b.wb_b.abs() == 9 {
                let k = self.eval_from(from, Params { wb: Wb::Kelvin(5500), wb_r: 0, wb_b: 0, ..b })?;
                let k = self.tune(Knob::Kelvin, k)?;
                self.wb_jacobian = None;
                self.tune_wb(k)?;
            }
        }

        self.set_phase("Tone");
        for knob in [Knob::Exposure, Knob::Highlight, Knob::Shadow] {
            self.tune(knob, self.best.unwrap())?;
        }
        if !mono {
            self.set_phase("Colour");
            self.tune(Knob::Color, self.best.unwrap())?;
            let levels = [ChromeLevel::Off, ChromeLevel::Weak, ChromeLevel::Strong];
            self.try_values(&levels, |p, v| Params { chrome: v, ..p.clone() })?;
            self.try_values(&levels, |p, v| Params { blue: v, ..p.clone() })?;
            self.tune_wb(self.best.unwrap())?;
        }
        if self.opts.dr_options.len() > 1 {
            self.set_phase("Dynamic range");
            let opts = self.opts.dr_options.clone();
            self.try_values(&opts, |p, v| Params { dr: v, ..p.clone() })?;
        }

        self.set_phase("Refine");
        let knobs: &[Knob] = if mono {
            &[Knob::Exposure, Knob::Highlight, Knob::Shadow]
        } else {
            &[Knob::Exposure, Knob::WbB, Knob::WbR, Knob::Highlight, Knob::Shadow, Knob::Color, Knob::Kelvin]
        };
        for pass in 0..4 {
            if !self.polish(knobs, pass == 0)? {
                break;
            }
        }
        Ok(())
    }
}

fn solve2(j: [[f32; 2]; 2], rhs: [f32; 2]) -> Option<(f32, f32)> {
    let det = j[0][0] * j[1][1] - j[0][1] * j[1][0];
    if det.abs() < 1e-3 {
        return None;
    }
    let x = (rhs[0] * j[1][1] - j[0][1] * rhs[1]) / det;
    let y = (j[0][0] * rhs[1] - j[1][0] * rhs[0]) / det;
    (x.is_finite() && y.is_finite()).then_some((x, y))
}

/// Search for the settings that best reproduce `reference`.
pub fn search(
    reference: &Reference,
    opts: &SearchOptions,
    stop: &AtomicBool,
    render: &mut RenderFn,
    on_event: &mut EventFn,
) -> Result<SearchResult, String> {
    let mut s = Search {
        reference,
        opts,
        stop,
        render,
        on_event,
        evals: Vec::new(),
        seen: HashMap::new(),
        best: None,
        phase: String::new(),
        renders: 0,
        wb_jacobian: None,
        slopes: HashMap::new(),
        failed: HashMap::new(),
    };
    let (stopped, budget_exhausted, error) = match s.run() {
        Ok(()) | Err(Halt::Done) => (false, false, None),
        Err(Halt::Stopped) => (true, false, None),
        Err(Halt::Budget) => (false, true, None),
        Err(Halt::Error(e)) => (true, false, Some(e)),
    };
    let Some(best) = s.best else {
        return Err(error.unwrap_or_else(|| "stopped before the first render".into()));
    };
    Ok(SearchResult { best: s.evals[best].clone(), renders: s.renders, stopped, budget_exhausted, error })
}

// ---------------------------------------------------------------------------
// Simulated camera (tests and FXRAW_STUB_CAMERA)
// ---------------------------------------------------------------------------

/// A rough, deterministic stand-in for the camera's development: every knob
/// moves the picture in the same direction the real one does. Used to test
/// the search without a camera; not an emulation of Fujifilm's processing.
pub fn simulate(base: &RgbImage, s: &RecipeSettings) -> RgbImage {
    use FilmSimulation::*;
    let sim = s.film_sim.unwrap_or(Provia);
    // (saturation, contrast, warmth, green tint, hue twist in degrees)
    let (sat, con, warm, tint, twist) = match sim {
        Provia => (1.0, 1.0, 0.0, 0.0, 0.0),
        Velvia => (1.35, 1.12, 0.01, 0.0, -4.0),
        Astia => (1.1, 0.95, 0.01, 0.0, -2.0),
        ClassicChrome => (0.75, 1.08, -0.01, 0.01, 5.0),
        ClassicNeg => (0.85, 1.15, -0.03, 0.03, 10.0),
        ProNegHi => (0.9, 1.05, 0.0, 0.0, 2.0),
        ProNegStd => (0.85, 0.92, 0.0, 0.0, 2.0),
        Eterna => (0.7, 0.85, 0.0, 0.01, 7.0),
        EternaBleachBypass => (0.45, 1.2, 0.0, 0.0, 0.0),
        NostalgicNeg => (0.9, 0.95, 0.05, -0.01, -8.0),
        RealaAce => (1.0, 1.02, 0.0, 0.0, 0.0),
        _ => (0.0, 1.05, 0.0, 0.0, 0.0),
    };
    let (tw_s, tw_c) = (twist as f32).to_radians().sin_cos();
    let (mut wr, mut wb) = match (s.white_balance.as_deref(), s.wb_temp) {
        (Some("daylight"), _) => (1.04, 0.95),
        (Some("temperature"), Some(k)) => {
            let w = (k as f32 - 5500.0) / 1000.0 * 0.05;
            (1.0 + w, 1.0 - w)
        }
        _ => (1.0, 1.0),
    };
    wr *= 1.0 + 0.02 * s.wb_shift_r.unwrap_or(0) as f32 + warm;
    wb *= 1.0 + 0.025 * s.wb_shift_b.unwrap_or(0) as f32 - warm;
    let wg = 1.0 + tint;
    let gain = 2f32.powf(s.exposure_comp.unwrap_or(0) as f32 / 1000.0);
    let hl = s.highlight.unwrap_or(0.0);
    let sh = s.shadow.unwrap_or(0.0);
    let sat = sat * (1.0 + 0.07 * s.color.unwrap_or(0) as f32);
    let chrome = match s.chrome_effect {
        Some(ChromeLevel::Weak) => 1.0,
        Some(ChromeLevel::Strong) => 2.0,
        _ => 0.0,
    };
    let blue = match s.chrome_blue {
        Some(ChromeLevel::Weak) => 1.0,
        Some(ChromeLevel::Strong) => 2.0,
        _ => 0.0,
    };
    let dr = match s.dynamic_range {
        Some(400) => 0.25,
        Some(200) => 0.12,
        _ => 0.0,
    };
    let lut = &*SRGB_TO_LINEAR;
    let smooth = |e0: f32, e1: f32, x: f32| {
        let t = ((x - e0) / (e1 - e0)).clamp(0.0, 1.0);
        t * t * (3.0 - 2.0 * t)
    };
    let mut out = base.clone();
    for p in out.pixels_mut() {
        let lin = [
            lut[p[0] as usize] * wr * gain,
            lut[p[1] as usize] * wg * gain,
            lut[p[2] as usize] * wb * gain,
        ];
        let mut c = lin.map(|v| v.max(0.0).powf(1.0 / 2.2));
        for v in c.iter_mut() {
            let mut y = 0.5 + (*v - 0.5) * con;
            y -= dr * y.powi(4) * 0.5;
            y += 0.035 * hl * smooth(0.5, 1.0, y) * (1.0 - smooth(0.9, 1.1, y));
            y -= 0.035 * sh * (1.0 - smooth(0.0, 0.5, y)) * smooth(-0.1, 0.1, y);
            *v = y;
        }
        let luma = 0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2];
        let mut cs = [0f32; 3];
        let satness = (c.iter().cloned().fold(f32::MIN, f32::max) - c.iter().cloned().fold(f32::MAX, f32::min))
            .clamp(0.0, 1.0);
        let is_blue = c[2] > c[0] && c[2] > c[1];
        for i in 0..3 {
            let mut k = sat;
            if is_blue {
                k *= 1.0 + 0.08 * blue;
            }
            cs[i] = (luma + (c[i] - luma) * k) * (1.0 - 0.05 * chrome * satness);
        }
        // Hue twist (rotation in the YIQ chroma plane).
        let yy = 0.299 * cs[0] + 0.587 * cs[1] + 0.114 * cs[2];
        let ii = 0.596 * cs[0] - 0.274 * cs[1] - 0.322 * cs[2];
        let qq = 0.211 * cs[0] - 0.523 * cs[1] + 0.312 * cs[2];
        let (ii, qq) = (ii * tw_c - qq * tw_s, ii * tw_s + qq * tw_c);
        let cs = [
            yy + 0.956 * ii + 0.621 * qq,
            yy - 0.272 * ii - 0.647 * qq,
            yy - 1.106 * ii + 1.703 * qq,
        ];
        for (i, v) in cs.iter().enumerate() {
            p[i] = (v.clamp(0.0, 1.0) * 255.0).round() as u8;
        }
    }
    out
}

pub fn encode_jpeg(img: &RgbImage, quality: u8) -> Vec<u8> {
    let mut buf = Vec::new();
    let _ = image::codecs::jpeg::JpegEncoder::new_with_quality(&mut buf, quality).encode_image(img);
    buf
}

/// Downscale for previews (fast, box-filtered).
pub fn thumbnail(img: &RgbImage, max_w: u32) -> RgbImage {
    if img.width() <= max_w {
        return img.clone();
    }
    let h = (img.height() as f64 * max_w as f64 / img.width() as f64).round() as u32;
    image::imageops::thumbnail(img, max_w, h.max(1))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn delta_e2000_matches_reference_values() {
        // Sharma, Wu & Dalal (2005) test data.
        let cases = [
            ([50.0, 2.6772, -79.7751], [50.0, 0.0, -82.7485], 2.0425),
            ([50.0, 3.1571, -77.2803], [50.0, 0.0, -82.7485], 2.8615),
            ([50.0, 2.5, 0.0], [73.0, 25.0, -18.0], 27.1492),
            ([60.2574, -34.0099, 36.2677], [60.4626, -34.1751, 39.4387], 1.2644),
            ([22.7233, 20.0904, -46.694], [23.0331, 14.973, -42.5619], 2.0373),
        ];
        for (a, b, want) in cases {
            let got = delta_e2000(a, b);
            assert!((got - want).abs() < 1e-3, "{a:?} vs {b:?}: {got} != {want}");
        }
    }

    #[test]
    fn lab_of_white_and_grey() {
        let w = linear_to_lab([1.0, 1.0, 1.0]);
        assert!((w[0] - 100.0).abs() < 0.01 && w[1].abs() < 0.01 && w[2].abs() < 0.01);
        let g = linear_to_lab([0.18, 0.18, 0.18]);
        assert!((g[0] - 49.5).abs() < 0.2);
    }

    #[test]
    fn rotation_and_unrotate_agree() {
        let img = Img { w: 3, h: 2, px: (0..6).map(|i| [i as f32, 0.0, 0.0]).collect() };
        for k in 0..4u8 {
            let r = img.rotate(k);
            for y in 0..r.h {
                for x in 0..r.w {
                    let (u, v) = ((x as f32 + 0.5) / r.w as f32, (y as f32 + 0.5) / r.h as f32);
                    let (ou, ov) = unrotate(k, u, v);
                    let (ox, oy) = ((ou * img.w as f32) as usize, (ov * img.h as f32) as usize);
                    assert_eq!(r.px[y * r.w + x], img.px[oy * img.w + ox], "k={k} at {x},{y}");
                }
            }
        }
    }

    fn fixture_frame() -> RgbImage {
        let raf = std::fs::read(concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures/DSCF6556.RAF")).unwrap();
        let jpeg = crate::fuji::embedded_jpeg(&raf).unwrap();
        let img = image::load_from_memory(jpeg).unwrap().to_rgb8();
        thumbnail(&img, 480)
    }

    /// A "scan": a different crop of the scene, scaled, turned upright,
    /// with a black film border.
    fn fake_scan(rendered: &RgbImage) -> RgbImage {
        let (w, h) = rendered.dimensions();
        let crop = image::imageops::crop_imm(rendered, w / 10, h / 8, w * 3 / 4, h * 3 / 4).to_image();
        let small = image::imageops::resize(&crop, 300, 200, image::imageops::FilterType::Triangle);
        let mut framed = RgbImage::from_pixel(320, 220, image::Rgb([8, 8, 8]));
        image::imageops::overlay(&mut framed, &small, 10, 10);
        image::imageops::rotate270(&framed)
    }

    #[test]
    fn aligns_a_cropped_rotated_scan() {
        let frame = fixture_frame();
        let target = Params::neutral(FilmSimulation::ClassicNeg, 100);
        let scan = fake_scan(&simulate(&frame, &target.to_settings()));
        let a = align(&scan, &frame);
        assert!(a.confident, "{a:?}");
        assert_eq!(a.rot90, 1, "{a:?}");
        // Crop covers x 0.1..0.85, y 0.125..0.875 of the frame; the scan is
        // that plus a border, so ~0.8 of the width wide.
        assert!((a.cx - 0.475).abs() < 0.03, "{a:?}");
        assert!((a.cy - 0.5).abs() < 0.03, "{a:?}");
        assert!((a.scale - 0.8).abs() < 0.06, "{a:?}");
    }

    #[test]
    fn search_recovers_simulated_settings() {
        let frame = fixture_frame();
        let mut target = Params::neutral(FilmSimulation::NostalgicNeg, 100);
        target.wb_r = 3;
        target.wb_b = -4;
        target.ev_thirds = 2;
        target.highlight = -1.0;
        target.color = 2;
        target.chrome = ChromeLevel::Strong;
        let scan = fake_scan(&simulate(&frame, &target.to_settings()));
        let reference = Reference::new(&scan, &frame);

        // The neutral starting point is clearly off.
        let start = reference.measure(&simulate(&frame, &Params::neutral(FilmSimulation::Provia, 100).to_settings()));
        assert!(start.loss > 4.0, "start loss {}", start.loss);

        let stop = AtomicBool::new(false);
        let opts = SearchOptions { max_renders: 70, good_enough: 0.6, ..Default::default() };
        let mut render = |s: &RecipeSettings| Ok(Rendered { image: simulate(&frame, s), jpeg: Vec::new() });
        let mut log = Vec::new();
        let mut on_event = |e: Event| {
            if let Event::Evaluated { eval, .. } = e {
                log.push(format!("{:>2} {:<16} {:<40} {:.2}", eval.n, eval.phase, eval.label, eval.m.loss));
            }
        };
        let res = search(&reference, &opts, &stop, &mut render, &mut on_event).unwrap();
        let p = &res.best.params;
        println!("{}", log.join("\n"));
        println!("best {:?} loss {:.2}", p, res.best.m.loss);
        // Different WB combinations can be equivalent, so judge the outcome:
        // the right look, the right exposure, and a near-invisible difference.
        assert!(res.best.m.loss < 1.0, "loss {}", res.best.m.loss);
        assert!(res.renders <= 60, "renders {}", res.renders);
        assert_eq!(p.film_sim, FilmSimulation::NostalgicNeg);
        assert!((p.ev_thirds - 2).abs() <= 1, "{p:?}");
    }
}
