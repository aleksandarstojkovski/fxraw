#![allow(dead_code)]

use crate::profile::{ChromeLevel, FilmSimulation, GrainEffect, GrainSize, RecipeSettings};
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};
use std::sync::{Arc, LazyLock, RwLock};

static RECIPES_JSON: &str = include_str!("../data/recipes.json");

/// Recipes compiled into the binary (data/recipes.json).
static BUILTIN: LazyLock<Vec<Recipe>> =
    LazyLock::new(|| serde_json::from_str(RECIPES_JSON).expect("invalid recipes.json"));

/// Built-in recipes followed by the user's own (`my_recipes.json`). Replaced
/// as a whole when the user saves or deletes one, so readers keep a
/// consistent snapshot.
static ALL: LazyLock<RwLock<Arc<Vec<Recipe>>>> = LazyLock::new(|| {
    let mine = my_recipes_path().map(|p| load_mine(&p)).unwrap_or_default();
    RwLock::new(Arc::new(combine(mine)))
});

#[derive(Debug, Clone, PartialEq, Deserialize, Serialize)]
pub struct Recipe {
    pub name: String,
    pub slug: String,
    pub film_sim: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub highlight: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub shadow: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub color: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub noise_reduction: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub sharpness: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub clarity: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub grain: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub grain_size: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub chrome_effect: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub chrome_blue: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub dynamic_range: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub white_balance: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub wb_temp: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub wb_shift_r: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub wb_shift_b: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub iso: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub exposure_comp: Option<String>,
    /// Where the recipe comes from (site, post, author).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub source: Option<RecipeSource>,
    /// Saved by the user in `my_recipes.json` (not part of the file format).
    #[serde(skip)]
    pub mine: bool,
}

#[derive(Debug, Clone, PartialEq, Deserialize, Serialize)]
pub struct RecipeSource {
    pub name: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub url: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub author: Option<String>,
}

impl RecipeSource {
    /// "Reddit r/fujifilm · u/indigo1917"
    pub fn label(&self) -> String {
        match &self.author {
            Some(a) => format!("{} · {a}", self.name),
            None => self.name.clone(),
        }
    }
}

impl Recipe {
    fn parse_film_sim(&self) -> Option<FilmSimulation> {
        match self.film_sim.as_str() {
            "provia" => Some(FilmSimulation::Provia),
            "velvia" => Some(FilmSimulation::Velvia),
            "astia" => Some(FilmSimulation::Astia),
            "classic-chrome" => Some(FilmSimulation::ClassicChrome),
            "classic-neg" => Some(FilmSimulation::ClassicNeg),
            "pro-neg-hi" => Some(FilmSimulation::ProNegHi),
            "pro-neg-std" => Some(FilmSimulation::ProNegStd),
            "eterna" => Some(FilmSimulation::Eterna),
            "eterna-bleach-bypass" => Some(FilmSimulation::EternaBleachBypass),
            "acros" => Some(FilmSimulation::Acros),
            "acros-ye" => Some(FilmSimulation::AcrosYe),
            "acros-r" => Some(FilmSimulation::AcrosR),
            "acros-g" => Some(FilmSimulation::AcrosG),
            "monochrome" => Some(FilmSimulation::Monochrome),
            "monochrome-ye" => Some(FilmSimulation::MonochromeYe),
            "monochrome-r" => Some(FilmSimulation::MonochromeR),
            "monochrome-g" => Some(FilmSimulation::MonochromeG),
            "sepia" => Some(FilmSimulation::Sepia),
            "nostalgic-neg" => Some(FilmSimulation::NostalgicNeg),
            "reala-ace" => Some(FilmSimulation::RealaAce),
            _ => None,
        }
    }

    fn parse_grain(&self) -> Option<GrainEffect> {
        match self.grain.as_deref() {
            Some("off") => Some(GrainEffect::Off),
            Some("weak") => Some(GrainEffect::Weak),
            Some("strong") => Some(GrainEffect::Strong),
            _ => None,
        }
    }

    fn parse_grain_size(&self) -> Option<GrainSize> {
        match self.grain_size.as_deref() {
            Some("small") => Some(GrainSize::Small),
            Some("large") => Some(GrainSize::Large),
            _ => None,
        }
    }

    fn parse_chrome(val: &Option<String>) -> Option<ChromeLevel> {
        match val.as_deref() {
            Some("off") => Some(ChromeLevel::Off),
            Some("weak") => Some(ChromeLevel::Weak),
            Some("strong") => Some(ChromeLevel::Strong),
            _ => None,
        }
    }

    pub fn to_settings(&self) -> RecipeSettings {
        RecipeSettings {
            film_sim: self.parse_film_sim(),
            grain: self.parse_grain(),
            grain_size: self.parse_grain_size(),
            highlight: self.highlight.map(|v| v as f32),
            shadow: self.shadow.map(|v| v as f32),
            color: self.color.map(|v| v as i32),
            sharpness: self.sharpness.map(|v| v as i32),
            noise_reduction: self.noise_reduction.map(|v| v as i32),
            clarity: self.clarity.map(|v| v as i32),
            white_balance: self.white_balance.clone(),
            wb_temp: self.wb_temp,
            wb_shift_r: self.wb_shift_r.map(|v| v as i32),
            wb_shift_b: self.wb_shift_b.map(|v| v as i32),
            chrome_effect: Self::parse_chrome(&self.chrome_effect),
            chrome_blue: Self::parse_chrome(&self.chrome_blue),
            dynamic_range: self.dynamic_range,
            exposure_comp: None,
        }
    }
}

/// Every recipe: built-in first, then the user's own.
pub fn all() -> Arc<Vec<Recipe>> {
    ALL.read().unwrap().clone()
}

pub fn find(query: &str) -> Option<Recipe> {
    let q = query.to_lowercase();
    let all = all();
    // Exact slug match first
    if let Some(r) = all.iter().find(|r| r.slug == q) {
        return Some(r.clone());
    }
    // Substring match on slug or name
    all.iter()
        .find(|r| r.slug.contains(&q) || r.name.to_lowercase().contains(&q))
        .cloned()
}

pub fn list_recipes() {
    let all = all();
    let (mine, builtin): (Vec<&Recipe>, Vec<&Recipe>) = all.iter().partition(|r| r.mine);
    println!("{} built-in recipes (X100VI compatible):\n", builtin.len());
    print_rows(&builtin);
    if !mine.is_empty() {
        let path = my_recipes_path().map(|p| p.display().to_string()).unwrap_or_default();
        println!("\n{} recipe(s) of your own ({path}):\n", mine.len());
        print_rows(&mine);
    }
}

fn print_rows(recipes: &[&Recipe]) {
    for r in recipes {
        let dr = match r.dynamic_range {
            Some(v) => format!("DR{v}"),
            None => "auto".into(),
        };
        let wb = match (r.white_balance.as_deref(), r.wb_temp) {
            (Some("temperature"), Some(t)) => format!("{t}K"),
            (Some(wb), _) => wb.to_string(),
            (None, _) => "auto".into(),
        };
        println!(
            "  {:<35} {:<22} {:<6} {}",
            r.slug, r.film_sim, dr, wb
        );
    }
}

pub fn show_recipe(r: &Recipe) {
    println!("{}", r.name);
    println!("{}", "─".repeat(r.name.len()));

    let row = |label: &str, val: String| {
        println!("  {:<18} {}", label, val);
    };

    row("Slug", r.slug.clone());
    row("Film Simulation", r.film_sim.clone());
    row("Dynamic Range", match r.dynamic_range {
        Some(dr) => format!("DR{dr}"),
        None => "auto".into(),
    });

    // Grain: "weak / small" or "off"
    let grain = match r.grain.as_deref() {
        Some(g) => {
            let size = r.grain_size.as_deref().unwrap_or("small");
            format!("{g} / {size}")
        }
        None => "off".into(),
    };
    row("Grain", grain);

    // Color Chrome + Blue on one line
    let chrome = r.chrome_effect.as_deref().unwrap_or("off");
    let chrome_b = r.chrome_blue.as_deref().unwrap_or("off");
    row("Color Effect", format!("chrome:{chrome}  blue:{chrome_b}"));

    // WB on one line: base + R/B shifts
    let wb_base = match (r.white_balance.as_deref(), r.wb_temp) {
        (Some("temperature"), Some(t)) => format!("{t}K"),
        (Some(wb), _) => wb.to_string(),
        (None, _) => "auto".into(),
    };
    let shift_r = r.wb_shift_r.map(|v| format!("{v:+}")).unwrap_or("0".into());
    let shift_b = r.wb_shift_b.map(|v| format!("{v:+}")).unwrap_or("0".into());
    row("White Balance", format!("{wb_base}  R:{shift_r} B:{shift_b}"));

    let fmt = |v: f64| -> String {
        if v == 0.0 { "0".into() } else { format!("{v:+}") }
    };

    // Tone: highlight & shadow on one line
    let hl = r.highlight.map(|v| fmt(v)).unwrap_or("0".into());
    let sh = r.shadow.map(|v| fmt(v)).unwrap_or("0".into());
    row("Tone", format!("highlight:{hl}  shadow:{sh}"));

    let fmti = |v: f64| -> String {
        if v == 0.0 { "0".into() } else { format!("{:+}", v as i32) }
    };
    row("Color", r.color.map(|v| fmti(v)).unwrap_or("0".into()));
    row("Sharpness", r.sharpness.map(|v| fmti(v)).unwrap_or("0".into()));
    row("NR", r.noise_reduction.map(|v| fmti(v)).unwrap_or("0".into()));
    row("Clarity", r.clarity.map(|v| fmti(v)).unwrap_or("0".into()));
    row("ISO", r.iso.clone().unwrap_or("auto".into()));
    row("Exposure", r.exposure_comp.clone().unwrap_or("0".into()));
    if let Some(src) = &r.source {
        row("Source", src.label());
        if let Some(url) = &src.url {
            row("", url.clone());
        }
    }
}

// ---------------------------------------------------------------------------
// The user's own recipes (~/.config/fxraw/my_recipes.json)
// ---------------------------------------------------------------------------

/// Why a recipe could not be saved.
#[derive(Debug, PartialEq)]
pub enum SaveError {
    /// Bad name or out-of-range setting.
    Invalid(String),
    /// The name maps to a built-in recipe's slug.
    Builtin(String),
    /// One of the user's recipes already has this slug (pass `overwrite`).
    Exists(String),
    Io(String),
}

impl std::fmt::Display for SaveError {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match self {
            Self::Invalid(m) | Self::Io(m) => f.write_str(m),
            Self::Builtin(s) => write!(f, "\"{s}\" is the name of a built-in recipe, pick another name"),
            Self::Exists(s) => write!(f, "you already have a recipe called \"{s}\""),
        }
    }
}

/// `$FXRAW_MY_RECIPES`, else `~/.config/fxraw/my_recipes.json` in the home of
/// the user who ran the command (the invoking user under `sudo`).
pub fn my_recipes_path() -> Option<PathBuf> {
    if let Some(p) = std::env::var_os("FXRAW_MY_RECIPES") {
        return Some(PathBuf::from(p));
    }
    let sudo_home = std::env::var("SUDO_USER")
        .ok()
        .filter(|u| !u.is_empty() && u != "root")
        .map(|u| PathBuf::from(if cfg!(target_os = "macos") { "/Users" } else { "/home" }).join(u))
        .filter(|h| h.is_dir());
    let home = sudo_home.or_else(|| std::env::var_os("HOME").map(PathBuf::from))?;
    Some(home.join(".config").join("fxraw").join("my_recipes.json"))
}

fn combine(mine: Vec<Recipe>) -> Vec<Recipe> {
    let mut all = BUILTIN.clone();
    all.extend(mine);
    all
}

/// Read the user's recipes. A missing file means none; a broken one is
/// reported and skipped so the built-in recipes still work.
fn load_mine(path: &Path) -> Vec<Recipe> {
    let text = match std::fs::read_to_string(path) {
        Ok(t) => t,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Vec::new(),
        Err(e) => {
            eprintln!("Warning: could not read {}: {e}", path.display());
            return Vec::new();
        }
    };
    let list: Vec<Recipe> = match serde_json::from_str(&text) {
        Ok(l) => l,
        Err(e) => {
            eprintln!("Warning: ignoring {}: {e}", path.display());
            return Vec::new();
        }
    };
    list.into_iter()
        .filter(|r| {
            let clash = is_builtin(&r.slug);
            if clash {
                eprintln!("Warning: {}: \"{}\" clashes with a built-in recipe, skipped", path.display(), r.slug);
            }
            !clash
        })
        .map(|mut r| {
            r.mine = true;
            r
        })
        .collect()
}

fn is_builtin(slug: &str) -> bool {
    BUILTIN.iter().any(|r| r.slug == slug)
}

/// "Essenza + grana forte!" -> "essenza-grana-forte"
pub fn slugify(name: &str) -> String {
    let mut slug = String::new();
    for c in name.trim().to_lowercase().chars() {
        if c.is_ascii_alphanumeric() {
            slug.push(c);
        } else if !slug.is_empty() && !slug.ends_with('-') {
            slug.push('-');
        }
    }
    slug.trim_end_matches('-').to_string()
}

const WB_MODES: &[&str] = &[
    "auto", "auto-white", "auto-ambience", "daylight", "shade", "incandescent",
    "fluorescent-1", "fluorescent-2", "fluorescent-3", "temperature", "underwater",
];

impl Recipe {
    /// Check every setting is one the camera accepts (X-Trans V ranges).
    pub fn validate(&self) -> Result<(), String> {
        if self.parse_film_sim().is_none() {
            return Err(format!("unknown film simulation \"{}\"", self.film_sim));
        }
        if let Some(dr) = self.dynamic_range {
            if ![100, 200, 400].contains(&dr) {
                return Err(format!("dynamic range must be 100, 200 or 400, not {dr}"));
            }
        }
        let one_of = |what: &str, v: &Option<String>, allowed: &[&str]| match v.as_deref() {
            Some(x) if !allowed.contains(&x) => Err(format!("{what} must be one of {}, not \"{x}\"", allowed.join(", "))),
            _ => Ok(()),
        };
        one_of("grain", &self.grain, &["off", "weak", "strong"])?;
        one_of("grain size", &self.grain_size, &["small", "large"])?;
        one_of("color chrome", &self.chrome_effect, &["off", "weak", "strong"])?;
        one_of("chrome FX blue", &self.chrome_blue, &["off", "weak", "strong"])?;
        one_of("white balance", &self.white_balance, WB_MODES)?;
        if self.white_balance.as_deref() == Some("temperature") {
            match self.wb_temp {
                Some(k) if (2500..=10000).contains(&k) => {}
                _ => return Err("Kelvin white balance needs a temperature between 2500 and 10000".into()),
            }
        }
        let range = |what: &str, v: Option<f64>, lo: f64, hi: f64, step: f64| match v {
            Some(x) if x < lo || x > hi || ((x / step).round() * step - x).abs() > 1e-9 => {
                Err(format!("{what} must be between {lo:+} and {hi:+} in steps of {step}, not {x}"))
            }
            _ => Ok(()),
        };
        range("highlight", self.highlight, -2.0, 4.0, 0.5)?;
        range("shadow", self.shadow, -2.0, 4.0, 0.5)?;
        range("color", self.color, -4.0, 4.0, 1.0)?;
        range("sharpness", self.sharpness, -4.0, 4.0, 1.0)?;
        range("noise reduction", self.noise_reduction, -4.0, 4.0, 1.0)?;
        range("clarity", self.clarity, -5.0, 5.0, 1.0)?;
        range("WB shift R", self.wb_shift_r, -9.0, 9.0, 1.0)?;
        range("WB shift B", self.wb_shift_b, -9.0, 9.0, 1.0)?;
        Ok(())
    }
}

/// Insert or replace `recipe` in the user's list, after giving it the slug of
/// its name and checking it. Pure: the caller persists the list.
fn upsert(mine: &mut Vec<Recipe>, mut recipe: Recipe, overwrite: bool) -> Result<Recipe, SaveError> {
    recipe.name = recipe.name.trim().to_string();
    recipe.slug = slugify(&recipe.name);
    if recipe.slug.is_empty() {
        return Err(SaveError::Invalid("give the recipe a name (letters or digits)".into()));
    }
    recipe.validate().map_err(SaveError::Invalid)?;
    if is_builtin(&recipe.slug) {
        return Err(SaveError::Builtin(recipe.slug));
    }
    recipe.mine = true;
    match mine.iter().position(|r| r.slug == recipe.slug) {
        Some(_) if !overwrite => Err(SaveError::Exists(recipe.slug)),
        Some(i) => {
            mine[i] = recipe.clone();
            Ok(recipe)
        }
        None => {
            mine.push(recipe.clone());
            Ok(recipe)
        }
    }
}

/// Write the user's recipes atomically, creating the folder if needed. Under
/// `sudo` the folder and file are handed back to the invoking user.
fn write_mine(path: &Path, mine: &[Recipe]) -> Result<(), String> {
    let dir = path.parent().ok_or("invalid recipes path")?;
    // Create missing folders one level at a time so each new one can be chowned.
    let mut missing: Vec<&Path> = dir.ancestors().take_while(|d| !d.exists()).collect();
    missing.reverse();
    for d in missing {
        std::fs::create_dir(d).map_err(|e| format!("could not create {}: {e}", d.display()))?;
        crate::fuji::chown_to_sudo_user(&d.to_string_lossy());
    }
    let mut text = serde_json::to_string_pretty(mine).map_err(|e| e.to_string())?;
    text.push('\n');
    let tmp = path.with_extension("json.tmp");
    std::fs::write(&tmp, text)
        .and_then(|_| std::fs::rename(&tmp, path))
        .map_err(|e| format!("could not write {}: {e}", path.display()))?;
    crate::fuji::chown_to_sudo_user(&path.to_string_lossy());
    Ok(())
}

/// Change the user's recipes with `f`, persist them, and publish the new list.
fn update_mine<T>(f: impl FnOnce(&mut Vec<Recipe>) -> Result<T, SaveError>) -> Result<T, SaveError> {
    let path = my_recipes_path().ok_or_else(|| SaveError::Io("no home folder to save recipes in".into()))?;
    let mut all = ALL.write().unwrap();
    let mut mine: Vec<Recipe> = all.iter().filter(|r| r.mine).cloned().collect();
    let out = f(&mut mine)?;
    write_mine(&path, &mine).map_err(SaveError::Io)?;
    *all = Arc::new(combine(mine));
    Ok(out)
}

/// Save `recipe` as one of the user's own, under the slug of its name.
/// Returns the stored recipe (with its slug).
pub fn save_mine(recipe: Recipe, overwrite: bool) -> Result<Recipe, SaveError> {
    update_mine(|mine| upsert(mine, recipe, overwrite))
}

/// Delete one of the user's recipes. `Ok(false)` if there is none with that slug.
pub fn delete_mine(slug: &str) -> Result<bool, SaveError> {
    update_mine(|mine| {
        let before = mine.len();
        mine.retain(|r| r.slug != slug);
        Ok(mine.len() != before)
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn recipe(name: &str) -> Recipe {
        let mut r = BUILTIN.iter().find(|r| r.slug == "essenza").unwrap().clone();
        r.name = name.into();
        r.grain = Some("strong".into());
        r.grain_size = Some("large".into());
        r
    }

    #[test]
    fn slugify_names() {
        assert_eq!(slugify("Essenza + grana forte!"), "essenza-grana-forte");
        assert_eq!(slugify("  Reggie's Portra 2 "), "reggie-s-portra-2");
        assert_eq!(slugify("¿?"), "");
    }

    #[test]
    fn builtin_recipes_are_valid() {
        for r in BUILTIN.iter() {
            assert_eq!(r.validate(), Ok(()), "{}", r.slug);
        }
    }

    #[test]
    fn validate_rejects_out_of_range() {
        let mut r = recipe("x");
        r.highlight = Some(4.5);
        assert!(r.validate().is_err());
        r.highlight = Some(1.25);
        assert!(r.validate().is_err());
        r.highlight = Some(-1.5);
        assert_eq!(r.validate(), Ok(()));
        r.grain = Some("huge".into());
        assert!(r.validate().is_err());
        r.grain = Some("off".into());
        r.white_balance = Some("temperature".into());
        r.wb_temp = None;
        assert!(r.validate().is_err());
        r.wb_temp = Some(5600);
        assert_eq!(r.validate(), Ok(()));
    }

    #[test]
    fn upsert_rules() {
        let mut mine = Vec::new();
        let saved = upsert(&mut mine, recipe(" Essenza grana "), false).unwrap();
        assert_eq!((saved.slug.as_str(), saved.name.as_str(), saved.mine), ("essenza-grana", "Essenza grana", true));
        assert_eq!(upsert(&mut mine, recipe("Essenza Grana"), false), Err(SaveError::Exists("essenza-grana".into())));
        let mut changed = recipe("Essenza Grana");
        changed.color = Some(-1.0);
        upsert(&mut mine, changed, true).unwrap();
        assert_eq!((mine.len(), mine[0].color), (1, Some(-1.0)));
        assert_eq!(upsert(&mut mine, recipe("Essenza"), true), Err(SaveError::Builtin("essenza".into())));
        assert!(matches!(upsert(&mut mine, recipe("!!"), false), Err(SaveError::Invalid(_))));
    }

    #[test]
    fn write_and_load_roundtrip() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("new").join("my_recipes.json");
        let mut mine = Vec::new();
        upsert(&mut mine, recipe("Essenza grana"), false).unwrap();
        write_mine(&path, &mine).unwrap();
        let text = std::fs::read_to_string(&path).unwrap();
        assert!(!text.contains("\"mine\""), "flag is not part of the file format");
        let loaded = load_mine(&path);
        assert_eq!(loaded.len(), 1);
        assert!(loaded[0].mine);
        assert_eq!(loaded[0].grain.as_deref(), Some("strong"));
        assert!(load_mine(&dir.path().join("missing.json")).is_empty());
    }

    #[test]
    fn grain_off_is_applied() {
        let mut r = recipe("x");
        r.grain = Some("off".into());
        assert_eq!(r.to_settings().grain, Some(GrainEffect::Off));
    }
}
