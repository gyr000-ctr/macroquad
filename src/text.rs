//! Functions to load fonts and draw text.

use std::collections::HashMap;

use crate::{
    color::Color,
    get_context, get_quad_context,
    math::{vec3, Rect},
    texture::{Image, TextureHandle},
    Error,
};

use crate::color::WHITE;
use glam::vec2;

use std::sync::{Arc, Mutex};
pub(crate) mod atlas;
pub(crate) mod shaping;

use atlas::{Atlas, SpriteKey};
use shaping::{
    decode_utf8_lossy, hash_text, shape_line, split_lines, GlyphFont, ShapedLine, ShapeCacheKey,
    MAX_FONT_SIZE,
};

#[derive(Debug, Clone)]
pub(crate) struct CharacterInfo {
    pub offset_x: i32,
    pub offset_y: i32,
    pub advance: f32,
    pub sprite: SpriteKey,
}

/// Glyph rasterization keyed by OpenType glyph index rather than Unicode
/// codepoint. Complex shaping selects presentation-form glyphs (Arabic
/// joining forms, matras) that share one codepoint namespace but live at
/// distinct indices, so the legacy `(char, size)` cache cannot hold them.
#[derive(Debug, Clone)]
#[allow(dead_code)] // `advance` is read by future shaped-measure consumers
pub(crate) struct IndexedGlyphInfo {
    pub offset_x: i32,
    pub offset_y: i32,
    pub advance: f32,
    pub sprite: SpriteKey,
}

/// Process-wide font id used as part of the shaping cache key.
static FONT_ID_COUNTER: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(1);

/// TTF font loaded to GPU
#[derive(Clone)]
pub struct Font {
    font: Arc<fontdue::Font>,
    atlas: Arc<Mutex<Atlas>>,
    characters: Arc<Mutex<HashMap<(char, u16), CharacterInfo>>>,
    /// Rasterized glyphs addressed by `(glyph_index, pixel_size)`. Used by the
    /// complex-shaping pipeline only; the legacy `characters` map is untouched
    /// so `draw_text` output stays byte-for-byte identical.
    indexed_glyphs: Arc<Mutex<HashMap<(u16, u16), IndexedGlyphInfo>>>,
    /// Unique id of this font instance, stable across its `Arc` clones.
    id: u64,
    /// Fallback fonts consulted in order when this font lacks coverage.
    fallbacks: Arc<Mutex<Vec<Font>>>,
    /// Bumped whenever the fallback chain changes; invalidates shaping cache
    /// entries whose resolved chain may have changed.
    chain_generation: Arc<std::sync::atomic::AtomicU64>,
    /// Shaped layout cache keyed by (font, chain generation, pixel size, text).
    /// Entries are shared `Arc`s so drawing never re-shapes the same string
    /// and callers only clone the cheap handle, never the glyph vectors.
    shape_cache: Arc<Mutex<HashMap<ShapeCacheKey, Arc<Vec<ShapedLine>>>>>,
}

/// World space dimensions of the text, measured by "measure_text" function
#[derive(Debug, Default, Clone, Copy)]
pub struct TextDimensions {
    /// Distance from very left to very right of the rasterized text
    pub width: f32,
    /// Distance from the bottom to the top of the text.
    pub height: f32,
    /// Height offset from the baseline of the text.
    /// "draw_text(.., X, Y, ..)" will be rendered in a "Rect::new(X, Y - dimensions.offset_y, dimensions.width, dimensions.height)"
    /// For reference check "text_measures" example.
    pub offset_y: f32,
}

#[allow(dead_code)]
fn require_fn_to_be_send() {
    fn require_send<T: Send>() {}
    require_send::<Font>();
}

impl std::fmt::Debug for Font {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Font")
            .field("font", &"fontdue::Font")
            .finish()
    }
}

impl Font {
    pub(crate) fn load_from_bytes(atlas: Arc<Mutex<Atlas>>, bytes: &[u8]) -> Result<Font, Error> {
        Ok(Font {
            font: Arc::new(fontdue::Font::from_bytes(
                bytes,
                fontdue::FontSettings::default(),
            )?),
            characters: Arc::new(Mutex::new(HashMap::new())),
            indexed_glyphs: Arc::new(Mutex::new(HashMap::new())),
            atlas,
            id: FONT_ID_COUNTER.fetch_add(1, std::sync::atomic::Ordering::Relaxed),
            fallbacks: Arc::new(Mutex::new(Vec::new())),
            chain_generation: Arc::new(std::sync::atomic::AtomicU64::new(0)),
            shape_cache: Arc::new(Mutex::new(HashMap::new())),
        })
    }

    pub(crate) fn set_atlas(&mut self, atlas: Arc<Mutex<Atlas>>) {
        self.atlas = atlas;
    }

    pub(crate) fn set_characters(
        &mut self,
        characters: Arc<Mutex<HashMap<(char, u16), CharacterInfo>>>,
    ) {
        self.characters = characters;
    }

    #[allow(dead_code)] // read by tests and future fallback diagnostics
    pub(crate) fn id(&self) -> u64 {
        self.id
    }

    pub(crate) fn chain_generation(&self) -> u64 {
        self.chain_generation
            .load(std::sync::atomic::Ordering::Relaxed)
    }

    /// Adds `font` to the end of this font's fallback chain. While shaping
    /// complex scripts, each codepoint is looked up in the primary font first
    /// and then in the fallbacks in registration order; a codepoint covered by
    /// none of them renders as the replacement glyph.
    ///
    /// The fallback font's atlas is used for the glyphs it supplies, so mixing
    /// fonts with different filter modes keeps each glyph rendered with its own
    /// font's settings.
    pub fn add_fallback_font(&self, font: Font) {
        self.fallbacks.lock().unwrap().push(font);
        self.chain_generation
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    }

    /// Snapshot of the resolved fallback chain (primary font first).
    pub(crate) fn fallback_chain(&self) -> Vec<Font> {
        let mut chain = Vec::with_capacity(
            self.fallbacks.lock().unwrap().len() + 1,
        );
        chain.push(self.clone());
        chain.extend(self.fallbacks.lock().unwrap().iter().cloned());
        chain
    }

    pub(crate) fn ascent(&self, font_size: f32) -> f32 {
        self.font.horizontal_line_metrics(font_size).unwrap().ascent
    }

    pub(crate) fn descent(&self, font_size: f32) -> f32 {
        self.font
            .horizontal_line_metrics(font_size)
            .unwrap()
            .descent
    }

    pub(crate) fn cache_glyph(&self, character: char, size: u16) {
        if self.contains(character, size) {
            return;
        }

        let (metrics, bitmap) = self.font.rasterize(character, size as f32);

        let (width, height) = (metrics.width as u16, metrics.height as u16);

        let sprite = self.atlas.lock().unwrap().new_unique_id();
        self.atlas.lock().unwrap().cache_sprite(
            sprite,
            Image {
                bytes: bitmap
                    .iter()
                    .flat_map(|coverage| [255, 255, 255, *coverage])
                    .collect(),
                width,
                height,
            },
        );
        let advance = metrics.advance_width;

        let (offset_x, offset_y) = (metrics.xmin, metrics.ymin);

        let character_info = CharacterInfo {
            advance,
            offset_x,
            offset_y,
            sprite,
        };

        self.characters
            .lock()
            .unwrap()
            .insert((character, size), character_info);
    }

    pub(crate) fn get(&self, character: char, size: u16) -> Option<CharacterInfo> {
        self.characters
            .lock()
            .unwrap()
            .get(&(character, size))
            .cloned()
    }
    /// Returns whether the character has been cached
    pub(crate) fn contains(&self, character: char, size: u16) -> bool {
        self.characters
            .lock()
            .unwrap()
            .contains_key(&(character, size))
    }

    /// Rasterizes (if needed) the glyph at OpenType `index` for the complex
    /// shaping pipeline. Missing glyphs resolve to index 0 (`.notdef`), which
    /// every TrueType font is required to contain, so this never fails.
    pub(crate) fn cache_indexed_glyph(&self, index: u16, size: u16) -> IndexedGlyphInfo {
        if let Some(info) = self.indexed_glyphs.lock().unwrap().get(&(index, size)) {
            return info.clone();
        }

        let (metrics, bitmap) = self.font.rasterize_indexed(index, size as f32);
        let (width, height) = (metrics.width as u16, metrics.height as u16);

        let sprite = self.atlas.lock().unwrap().new_unique_id();
        self.atlas.lock().unwrap().cache_sprite(
            sprite,
            Image {
                bytes: bitmap
                    .iter()
                    .flat_map(|coverage| [255, 255, 255, *coverage])
                    .collect(),
                width,
                height,
            },
        );

        let info = IndexedGlyphInfo {
            offset_x: metrics.xmin,
            offset_y: metrics.ymin,
            advance: metrics.advance_width,
            sprite,
        };
        self.indexed_glyphs
            .lock()
            .unwrap()
            .insert((index, size), info.clone());
        info
    }

    pub(crate) fn contains_indexed(&self, index: u16, size: u16) -> bool {
        self.indexed_glyphs
            .lock()
            .unwrap()
            .contains_key(&(index, size))
    }

    /// Coverage check used by the fallback resolver.
    #[allow(dead_code)] // exercised by the shaping integration tests
    pub(crate) fn covers(&self, character: char) -> bool {
        self.font.has_glyph(character)
    }

    pub(crate) fn glyph_index(&self, character: char) -> u16 {
        self.font.lookup_glyph_index(character)
    }

    pub(crate) fn measure_text(
        &self,
        text: impl AsRef<str>,
        font_size: u16,
        font_scale_x: f32,
        font_scale_y: f32,
        mut glyph_callback: impl FnMut(f32),
    ) -> TextDimensions {
        let text = text.as_ref();

        if text.is_empty() {
            return TextDimensions::default();
        }

        let dpi_scaling = miniquad::window::dpi_scale();
        let font_size = (font_size as f32 * dpi_scaling).ceil() as u16;

        let mut width = 0.0;
        let mut min_y = f32::MAX;
        let mut max_y = f32::MIN;

        for character in text.chars() {
            if !self.contains(character, font_size) {
                self.cache_glyph(character, font_size);
            }

            let font_data = &self.characters.lock().unwrap()[&(character, font_size)];
            let offset_y = font_data.offset_y as f32 * font_scale_y;

            let atlas = self.atlas.lock().unwrap();
            let glyph = atlas.get(font_data.sprite).unwrap().rect;
            let advance = font_data.advance * font_scale_x;
            glyph_callback(advance);
            width += advance;
            min_y = min_y.min(offset_y);
            max_y = max_y.max(glyph.h * font_scale_y + offset_y);
        }

        TextDimensions {
            width: width / dpi_scaling,
            height: (max_y - min_y) / dpi_scaling,
            offset_y: max_y / dpi_scaling,
        }
    }

    /// Shapes one logical line of already UTF-8-decoded text, returning the
    /// cached layout for the whole multiline string when called through
    /// [`shape_text`]. Positions are in rasterized (pixel) units.
    fn shape_cached(&self, text: &str, font_size: u16) -> Arc<Vec<ShapedLine>> {
        let key = ShapeCacheKey {
            font_id: self.id,
            chain_generation: self.chain_generation(),
            font_size,
            text_hash: hash_text(text),
        };

        if let Some(lines) = self.shape_cache.lock().unwrap().get(&key) {
            return lines.clone();
        }

        let chain = self.fallback_chain();
        let resolved: Vec<&dyn GlyphFont> =
            chain.iter().map(|f| &*f.font as &dyn GlyphFont).collect();
        let lines: Vec<ShapedLine> = split_lines(text)
            .into_iter()
            .map(|line| shape_line(&resolved, &line, font_size as f32))
            .collect();
        let lines = Arc::new(lines);

        self.shape_cache
            .lock()
            .unwrap()
            .insert(key, lines.clone());
        lines
    }

    /// Public shaping entry point with font-size clamping. Gigantic sizes
    /// would allocate atlas images hundreds of megabytes large, so the size is
    /// capped at [`MAX_FONT_SIZE`] instead of crashing; sub-pixel sizes below
    /// one pixel rasterize at 1px and keep scaling via `font_scale`.
    pub(crate) fn shape_text(&self, text: &str, raw_font_size: u16) -> Arc<Vec<ShapedLine>> {
        let font_size = raw_font_size.clamp(1, MAX_FONT_SIZE);
        self.shape_cached(text, font_size)
    }
}

impl Font {
    /// List of ascii characters, may be helpful in combination with "populate_font_cache"
    pub fn ascii_character_list() -> Vec<char> {
        (0..255).filter_map(::std::char::from_u32).collect()
    }

    /// List of latin characters
    pub fn latin_character_list() -> Vec<char> {
        "qwertyuiopasdfghjklzxcvbnmQWERTYUIOPASDFGHJKLZXCVBNM1234567890!@#$%^&*(){}[].,:"
            .chars()
            .collect()
    }

    pub fn populate_font_cache(&self, characters: &[char], size: u16) {
        for character in characters {
            self.cache_glyph(*character, size);
        }
    }

    /// Sets the [FilterMode](https://docs.rs/miniquad/latest/miniquad/graphics/enum.FilterMode.html#) of this font's texture atlas.
    ///
    /// Use Nearest if you need integer-ratio scaling for pixel art, for example.
    ///
    /// # Example
    /// ```
    /// # use macroquad::prelude::*;
    /// # #[macroquad::main("test")]
    /// # async fn main() {
    /// let mut font = get_default_font();
    /// font.set_filter(FilterMode::Linear);
    /// # }
    /// ```
    pub fn set_filter(&mut self, filter_mode: miniquad::FilterMode) {
        self.atlas.lock().unwrap().set_filter(filter_mode);
    }

    // pub fn texture(&self) -> Texture2D {
    //     let font = get_context().fonts_storage.get_font(*self);

    //     font.font_texture
    // }
}

impl Default for Font {
    fn default() -> Self {
        get_default_font()
    }
}

/// Arguments for "draw_text_ex" function such as font, font_size etc
#[derive(Debug, Clone)]
pub struct TextParams<'a> {
    pub font: Option<&'a Font>,
    /// Base size for character height. The size in pixel used during font rasterizing.
    pub font_size: u16,
    /// The glyphs sizes actually drawn on the screen will be font_size * font_scale
    /// However with font_scale too different from 1.0 letters may be blurry
    pub font_scale: f32,
    /// Font X axis would be scaled by font_scale * font_scale_aspect
    /// and Y axis would be scaled by font_scale
    /// Default is 1.0
    pub font_scale_aspect: f32,
    /// Text rotation in radian
    /// Default is 0.0
    pub rotation: f32,
    pub color: Color,
    /// Enables the complex text shaping pipeline: script/direction
    /// segmentation, Arabic joining and ligatures, Devanagari matra
    /// reordering, RTL visual ordering, combining-mark anchoring, kerning and
    /// per-glyph font fallback.
    ///
    /// Defaults to `false`, in which case `draw_text_ex` takes the legacy
    /// codepoint-by-codepoint path with pixel-identical output to older
    /// macroquad versions.
    pub complex_text: bool,
}

impl<'a> Default for TextParams<'a> {
    fn default() -> TextParams<'a> {
        TextParams {
            font: None,
            font_size: 20,
            font_scale: 1.0,
            font_scale_aspect: 1.0,
            color: WHITE,
            rotation: 0.0,
            complex_text: false,
        }
    }
}

/// Load font from file with "path"
pub async fn load_ttf_font(path: &str) -> Result<Font, Error> {
    let bytes = crate::file::load_file(path)
        .await
        .map_err(|_| Error::FontError("The Font file couldn't be loaded"))?;

    load_ttf_font_from_bytes(&bytes[..])
}

/// Load font from bytes array, may be use in combination with include_bytes!
/// ```ignore
/// let font = load_ttf_font_from_bytes(include_bytes!("font.ttf"));
/// ```
pub fn load_ttf_font_from_bytes(bytes: &[u8]) -> Result<Font, Error> {
    let atlas = Arc::new(Mutex::new(Atlas::new(
        get_quad_context(),
        miniquad::FilterMode::Linear,
    )));

    let mut font = Font::load_from_bytes(atlas.clone(), bytes)?;

    font.populate_font_cache(&Font::ascii_character_list(), 15);

    let ctx = get_context();

    font.set_filter(ctx.default_filter_mode);

    Ok(font)
}

/// Draw text with given font_size
/// Returns text size
pub fn draw_text(
    text: impl AsRef<str>,
    x: f32,
    y: f32,
    font_size: f32,
    color: Color,
) -> TextDimensions {
    draw_text_ex(
        text,
        x,
        y,
        TextParams {
            font_size: font_size as u16,
            font_scale: 1.0,
            color,
            ..Default::default()
        },
    )
}

/// Draw text with custom params such as font, font size and font scale
/// Returns text size
pub fn draw_text_ex(text: impl AsRef<str>, x: f32, y: f32, params: TextParams) -> TextDimensions {
    let text = text.as_ref();

    if text.is_empty() {
        return TextDimensions::default();
    }

    if params.complex_text {
        return draw_text_shaped(text, x, y, &params);
    }

    let font = params
        .font
        .unwrap_or(&get_context().fonts_storage.default_font);

    let dpi_scaling = miniquad::window::dpi_scale();

    let rot = params.rotation;
    let font_scale_x = params.font_scale * params.font_scale_aspect;
    let font_scale_y = params.font_scale;
    let font_size = (params.font_size as f32 * dpi_scaling).ceil() as u16;
    let font_size_f32 = font_size as f32;

    let mut total_width = 0.0;
    let mut max_offset_y = f32::MIN;
    let mut min_offset_y = f32::MAX;

    let mut last_character = None;

    for character in text.chars() {
        if !font.contains(character, font_size) {
            font.cache_glyph(character, font_size);
        }

        let kerning_offset = last_character
            .and_then(|left| font.font.horizontal_kern(left, character, font_size_f32))
            .unwrap_or(0.0);
        last_character = Some(character);

        let char_data = &font.characters.lock().unwrap()[&(character, font_size)];
        let offset_x = char_data.offset_x as f32 * font_scale_x;
        let offset_y = char_data.offset_y as f32 * font_scale_y;

        let mut atlas = font.atlas.lock().unwrap();
        let glyph = atlas.get(char_data.sprite).unwrap().rect;
        let glyph_scaled_h = glyph.h * font_scale_y;

        min_offset_y = min_offset_y.min(offset_y);
        max_offset_y = max_offset_y.max(glyph_scaled_h + offset_y);

        let rot_cos = rot.cos();
        let rot_sin = rot.sin();
        let dest_x = (offset_x + total_width) * rot_cos + (glyph_scaled_h + offset_y) * rot_sin;
        let dest_y = (offset_x + total_width) * rot_sin + (-glyph_scaled_h - offset_y) * rot_cos;

        // Kerning shifts along the text-flow direction. At `rot = 0`
        // text flows along +X, so the offset goes into dest.x; at
        // 90° / 270° text flows along ±Y, so it needs to go into
        // dest.y instead. Multiplying by (rot_cos, rot_sin) routes
        // the offset into the correct axis for any rotation.
        let kerning_logical = kerning_offset / dpi_scaling;
        let dest = Rect::new(
            dest_x / dpi_scaling + x + kerning_logical * rot_cos,
            dest_y / dpi_scaling + y + kerning_logical * rot_sin,
            glyph.w / dpi_scaling * font_scale_x,
            glyph.h / dpi_scaling * font_scale_y,
        );

        total_width += char_data.advance * font_scale_x + kerning_offset;

        crate::texture::draw_texture_ex(
            &crate::texture::Texture2D {
                texture: TextureHandle::Unmanaged(atlas.texture()),
            },
            dest.x,
            dest.y,
            params.color,
            crate::texture::DrawTextureParams {
                dest_size: Some(vec2(dest.w, dest.h)),
                source: Some(glyph),
                rotation: rot,
                pivot: Some(vec2(dest.x, dest.y)),
                ..Default::default()
            },
        );
    }

    TextDimensions {
        width: total_width / dpi_scaling,
        height: (max_offset_y - min_offset_y) / dpi_scaling,
        offset_y: max_offset_y / dpi_scaling,
    }
}

/// Complex-shaping draw path. The shaped layout comes from the per-font
/// shaping cache (so repeated frames only clone an `Arc`), and glyph
/// rasterization goes through the indexed glyph cache plus each font's atlas.
/// No allocation happens per glyph: all vectors are reused across lines and
/// every draw call reuses the same `DrawTextureParams` stack value.
fn draw_text_shaped(text: &str, x: f32, y: f32, params: &TextParams) -> TextDimensions {
    let font = params
        .font
        .cloned()
        .unwrap_or_else(|| get_context().fonts_storage.default_font.clone());

    let dpi_scaling = miniquad::window::dpi_scale();
    let font_scale_x = params.font_scale * params.font_scale_aspect;
    let font_scale_y = params.font_scale;
    // Clamp before both shaping and rasterization. Without this, a huge
    // requested size would make fontdue allocate a multi-gigabyte bitmap per
    // glyph even though the layout itself is capped. The clamped raster is
    // still scaled on screen through `font_scale`, so the text keeps its
    // requested (blurry) size instead of exhausting memory.
    let raster_size =
        ((params.font_size as f32 * dpi_scaling).ceil() as u16).clamp(1, MAX_FONT_SIZE);
    let rot = params.rotation;

    let shaped = font.shape_text(text, raster_size);
    let chain = font.fallback_chain();

    // Effective font size in world units. When the requested size was
    // clamped, glyphs were rasterized at `raster_size`, so the line metrics
    // must derive from the same effective size to stay consistent with what
    // is actually drawn — otherwise a single oversized line would report a
    // height hundreds of times larger than its rendered extent.
    let effective_font_size = raster_size as f32 / dpi_scaling;

    // Line step in world units: the primary font's line gap, falling back to
    // the effective font size when the font exposes no line metrics.
    let line_step = font
        .font
        .horizontal_line_metrics(1.0)
        .map(|m| m.new_line_size)
        .unwrap_or(1.0)
        * effective_font_size
        * params.font_scale;

    let mut dimensions = TextDimensions::default();
    let mut line_y = 0.0_f32;

    for line in shaped.iter() {
        let line_dims = draw_shaped_line(
            line,
            &chain,
            x,
            y + line_y,
            raster_size,
            dpi_scaling,
            font_scale_x,
            font_scale_y,
            rot,
            params.color,
        );
        dimensions.width = dimensions.width.max(line_dims.width);
        dimensions.height += if line_step > 0.0 { line_step } else { line_dims.height };
        if dimensions.offset_y == 0.0 {
            dimensions.offset_y = line_dims.offset_y;
        }
        line_y += line_step;
    }

    dimensions
}

#[allow(clippy::too_many_arguments)]
fn draw_shaped_line(
    line: &ShapedLine,
    chain: &[Font],
    x: f32,
    y: f32,
    raster_size: u16,
    dpi_scaling: f32,
    font_scale_x: f32,
    font_scale_y: f32,
    rot: f32,
    color: Color,
) -> TextDimensions {
    let rot_cos = rot.cos();
    let rot_sin = rot.sin();
    let mut max_offset_y = f32::MIN;
    let mut min_offset_y = f32::MAX;

    for glyph in &line.glyphs {
        let slot = glyph.font_slot.min(chain.len() - 1);
        let slot_font = &chain[slot];

        // Missing codepoints were already mapped onto the replacement
        // character during shaping, so `glyph.char` always addresses a glyph
        // the slot font contains (its `.notdef` when nothing else covers it).
        let index = slot_font.glyph_index(glyph.char);

        if !slot_font.contains_indexed(index, raster_size) {
            slot_font.cache_indexed_glyph(index, raster_size);
        }
        let info = slot_font
            .indexed_glyphs
            .lock()
            .unwrap()
            .get(&(index, raster_size))
            .cloned()
            .expect("indexed glyph was just cached");

        let mut atlas_guard = slot_font.atlas.lock().unwrap();
        let sprite = atlas_guard.get(info.sprite).unwrap().rect;
        let texture_id = atlas_guard.texture();
        drop(atlas_guard);

        let offset_x = info.offset_x as f32 * font_scale_x;
        let offset_y = (info.offset_y as f32 - glyph.y_offset) * font_scale_y;
        let glyph_scaled_h = sprite.h * font_scale_y;

        min_offset_y = min_offset_y.min(offset_y);
        max_offset_y = max_offset_y.max(glyph_scaled_h + offset_y);

        let pen_x = (glyph.x * font_scale_x + offset_x) / dpi_scaling;
        let dest_x = pen_x * rot_cos + (glyph_scaled_h + offset_y) * rot_sin + x;
        let dest_y = pen_x * rot_sin + (-glyph_scaled_h - offset_y) * rot_cos + y;

        let dest = Rect::new(
            dest_x,
            dest_y,
            sprite.w / dpi_scaling * font_scale_x,
            sprite.h / dpi_scaling * font_scale_y,
        );

        crate::texture::draw_texture_ex(
            &crate::texture::Texture2D {
                texture: TextureHandle::Unmanaged(texture_id),
            },
            dest.x,
            dest.y,
            color,
            crate::texture::DrawTextureParams {
                dest_size: Some(vec2(dest.w, dest.h)),
                source: Some(sprite),
                rotation: rot,
                pivot: Some(vec2(dest.x, dest.y)),
                ..Default::default()
            },
        );
    }

    if line.glyphs.is_empty() {
        return TextDimensions {
            width: 0.0,
            height: 0.0,
            offset_y: 0.0,
        };
    }

    TextDimensions {
        width: line.width * font_scale_x / dpi_scaling,
        height: (max_offset_y - min_offset_y) / dpi_scaling,
        offset_y: max_offset_y / dpi_scaling,
    }
}

/// Like [`draw_text_ex`] but accepts raw bytes: invalid UTF-8 sequences are
/// rendered as the replacement glyph (U+FFFD) instead of producing garbled
/// output or panicking. Complex shaping is always enabled on this entry
/// point; newlines split the text into separate lines.
pub fn draw_text_lossy(
    bytes: &[u8],
    x: f32,
    y: f32,
    font_size: f32,
    color: Color,
) -> TextDimensions {
    let text: String = decode_utf8_lossy(bytes).into_iter().collect();
    draw_text_ex(
        &text,
        x,
        y,
        TextParams {
            font_size: font_size as u16,
            font_scale: 1.0,
            color,
            complex_text: true,
            ..Default::default()
        },
    )
}

/// Measures text shaped with the complex pipeline. The returned width
/// matches what [`draw_text_ex`] with `complex_text: true` would occupy,
/// including ligature, kerning and fallback-font advances.
pub fn measure_text_shaped(
    text: impl AsRef<str>,
    font: Option<&Font>,
    font_size: u16,
    font_scale: f32,
) -> TextDimensions {
    let font = font
        .cloned()
        .unwrap_or_else(|| get_context().fonts_storage.default_font.clone());
    if text.as_ref().is_empty() {
        return TextDimensions::default();
    }
    let dpi_scaling = miniquad::window::dpi_scale();
    // `shape_text` clamps internally; clamping here too keeps the requested
    // and effective size explicit and the cache key identical to the draw
    // path.
    let raster_size =
        ((font_size as f32 * dpi_scaling).ceil() as u16).clamp(1, MAX_FONT_SIZE);
    let shaped = font.shape_text(text.as_ref(), raster_size);

    let mut dimensions = TextDimensions::default();
    for line in shaped.iter() {
        let width = line.width * font_scale / dpi_scaling;
        dimensions.width = dimensions.width.max(width);
        // Height comes from the primary font's metrics; the shaped pipeline
        // cannot derive glyph extents without rasterizing.
        if let Some(metrics) = font.font.horizontal_line_metrics(raster_size as f32) {
            dimensions.height =
                dimensions.height.max((metrics.ascent - metrics.descent) / dpi_scaling);
            dimensions.offset_y = metrics.ascent / dpi_scaling;
        }
    }
    dimensions
}
/// If no line distance but a custom font is given, the fonts line gap will be used as line distance factor if it exists.
pub fn draw_multiline_text(
    text: impl AsRef<str>,
    x: f32,
    y: f32,
    font_size: f32,
    line_distance_factor: Option<f32>,
    color: Color,
) -> TextDimensions {
    draw_multiline_text_ex(
        text,
        x,
        y,
        line_distance_factor,
        TextParams {
            font_size: font_size as u16,
            font_scale: 1.0,
            color,
            ..Default::default()
        },
    )
}

/// Draw multiline text with the given line distance and custom params such as font, font size and font scale.
/// If no line distance but a custom font is given, the fonts newline size will be used as line distance factor if it exists, else default to font size.
pub fn draw_multiline_text_ex(
    text: impl AsRef<str>,
    mut x: f32,
    mut y: f32,
    line_distance_factor: Option<f32>,
    params: TextParams,
) -> TextDimensions {
    let line_distance = match line_distance_factor {
        Some(distance) => distance,
        None => {
            let mut font_line_distance = 0.0;
            let font = if let Some(font) = params.font {
                font
            } else {
                &get_default_font()
            };
            if let Some(metrics) = font.font.horizontal_line_metrics(1.0) {
                font_line_distance = metrics.new_line_size;
            }

            font_line_distance
        }
    };

    let mut dimensions = TextDimensions::default();
    let y_step = line_distance * params.font_size as f32 * params.font_scale;

    for line in text.as_ref().lines() {
        let line_dimensions = draw_text_ex(line, x, y, params.clone());
        x -= (line_distance * params.font_size as f32 * params.font_scale) * params.rotation.sin();
        y += (line_distance * params.font_size as f32 * params.font_scale) * params.rotation.cos();

        dimensions.width = f32::max(dimensions.width, line_dimensions.width);
        dimensions.height += y_step;

        if dimensions.offset_y == 0.0 {
            dimensions.offset_y = line_dimensions.offset_y;
        }
    }

    dimensions
}

/// Get the text center.
pub fn get_text_center(
    text: impl AsRef<str>,
    font: Option<&Font>,
    font_size: u16,
    font_scale: f32,
    rotation: f32,
) -> crate::Vec2 {
    let measure = measure_text(text, font, font_size, font_scale);

    let x_center = measure.width / 2.0 * rotation.cos() + measure.height / 2.0 * rotation.sin();
    let y_center = measure.width / 2.0 * rotation.sin() - measure.height / 2.0 * rotation.cos();

    crate::Vec2::new(x_center, y_center)
}

pub fn measure_text(
    text: impl AsRef<str>,
    font: Option<&Font>,
    font_size: u16,
    font_scale: f32,
) -> TextDimensions {
    let font = font.unwrap_or_else(|| &get_context().fonts_storage.default_font);

    font.measure_text(text, font_size, font_scale, font_scale, |_| {})
}

pub fn measure_multiline_text(
    text: &str,
    font: Option<&Font>,
    font_size: u16,
    font_scale: f32,
    line_distance_factor: Option<f32>,
) -> TextDimensions {
    let font = font.unwrap_or_else(|| &get_context().fonts_storage.default_font);
    let line_distance = match line_distance_factor {
        Some(distance) => distance,
        None => match font.font.horizontal_line_metrics(1.0) {
            Some(metrics) => metrics.new_line_size,
            None => 1.0,
        },
    };

    let mut dimensions = TextDimensions::default();
    let y_step = line_distance * font_size as f32 * font_scale;

    for line in text.lines() {
        let line_dimensions = font.measure_text(line, font_size, font_scale, font_scale, |_| {});

        dimensions.width = f32::max(dimensions.width, line_dimensions.width);
        dimensions.height += y_step;
        if dimensions.offset_y == 0.0 {
            dimensions.offset_y = line_dimensions.offset_y;
        }
    }

    dimensions
}

/// Converts word breaks to newlines wherever the text would otherwise exceed the given length.
pub fn wrap_text(
    text: &str,
    font: Option<&Font>,
    font_size: u16,
    font_scale: f32,
    maximum_line_length: f32,
) -> String {
    let font = font.unwrap_or_else(|| &get_context().fonts_storage.default_font);

    // This is always a bit too much memory, but it saves a lot of reallocations.
    let mut new_text =
        String::with_capacity(text.len() + text.chars().filter(|c| c.is_whitespace()).count());

    let mut current_word_start = 0usize;
    let mut current_word_end = 0usize;
    let mut characters = text.char_indices();
    let mut total_width = 0.0;
    let mut word_width = 0.0;

    font.measure_text(text, font_size, font_scale, font_scale, |mut width| {
        // It's impossible this is called more often than the text has characters.
        let (idx, c) = characters.next().unwrap();
        let mut keep_char = true;

        if c.is_whitespace() {
            new_text.push_str(&text[current_word_start..idx + c.len_utf8()]);
            current_word_start = idx + c.len_utf8();
            word_width = 0.0;
            keep_char = false;

            // If we would wrap, ignore the whitespace.
            if total_width + width > maximum_line_length {
                width = 0.0;
            }
        }

        // If a single word expands past the length limit, just break it up.
        if word_width + width > maximum_line_length {
            new_text.push_str(&text[current_word_start..current_word_end]);
            new_text.push('\n');
            current_word_start = current_word_end;
            total_width = 0.0;
            word_width = 0.0;
        }

        current_word_end = idx + c.len_utf8();
        if keep_char {
            word_width += width;
        }

        if c == '\n' {
            total_width = 0.0;
            word_width = 0.0;
            return;
        }

        total_width += width;

        if total_width > maximum_line_length {
            new_text.push('\n');
            total_width = word_width;
        }
    });

    new_text.push_str(&text[current_word_start..current_word_end]);

    new_text
}

pub(crate) struct FontsStorage {
    default_font: Font,
}

impl FontsStorage {
    pub(crate) fn new(ctx: &mut dyn miniquad::RenderingBackend) -> FontsStorage {
        let atlas = Arc::new(Mutex::new(Atlas::new(ctx, miniquad::FilterMode::Linear)));

        let default_font = Font::load_from_bytes(atlas, include_bytes!("ProggyClean.ttf")).unwrap();
        FontsStorage { default_font }
    }
}

/// Returns macroquads default font.
pub fn get_default_font() -> Font {
    let context = get_context();
    context.fonts_storage.default_font.clone()
}

/// Replaces macroquads default font with `font`.
pub fn set_default_font(font: Font) {
    let context = get_context();
    context.fonts_storage.default_font = font;
}

/// From given font size in world space gives
/// (font_size, font_scale and font_aspect) params to make rasterized font
/// looks good in currently active camera
pub fn camera_font_scale(world_font_size: f32) -> (u16, f32, f32) {
    let context = get_context();
    let (scr_w, scr_h) = miniquad::window::screen_size();
    let cam_space = context
        .projection_matrix()
        .inverse()
        .transform_vector3(vec3(2., 2., 0.));
    let (cam_w, cam_h) = (cam_space.x.abs(), cam_space.y.abs());

    let screen_font_size = world_font_size * scr_h / cam_h;

    let font_size = screen_font_size as u16;

    (font_size, cam_h / scr_h, scr_h / scr_w * cam_w / cam_h)
}
