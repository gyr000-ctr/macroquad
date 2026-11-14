use crate::{get_context, get_quad_context, math::Rect, texture::Image, Color};

use std::collections::HashMap;

#[derive(Debug, Clone, Copy)]
pub struct Sprite {
    pub rect: Rect,
}

/// Resizes `image` so neither dimension exceeds `max_size`, preserving aspect
/// ratio. Nearest-neighbor sampling is sufficient because the result is only
/// ever an emergency fallback for glyphs that should never have been
/// rasterized that large.
fn downscale_image(image: Image, max_size: u16) -> Image {
    let scale = max_size as f32 / image.width.max(image.height) as f32;
    let new_width = ((image.width as f32 * scale).round() as u16).max(1);
    let new_height = ((image.height as f32 * scale).round() as u16).max(1);

    let mut bytes = Vec::with_capacity(new_width as usize * new_height as usize * 4);
    for ny in 0..new_height {
        for nx in 0..new_width {
            let sx = (nx as u32 * image.width as u32 / new_width as u32) as usize;
            let sy = (ny as u32 * image.height as u32 / new_height as u32) as usize;
            let src = (sy * image.width as usize + sx) * 4;
            bytes.extend_from_slice(&image.bytes[src..src + 4]);
        }
    }
    Image {
        bytes,
        width: new_width,
        height: new_height,
    }
}

#[derive(Debug, Eq, PartialEq, Hash, Clone, Copy)]
pub enum SpriteKey {
    Texture(miniquad::TextureId),
    Id(u64),
}
pub struct Atlas {
    texture: miniquad::TextureId,
    image: Image,
    pub sprites: HashMap<SpriteKey, Sprite>,
    cursor_x: u16,
    cursor_y: u16,
    max_line_height: u16,

    pub dirty: bool,

    filter: miniquad::FilterMode,

    unique_id: u64,
}

/// Upper bound for the atlas dimensions. Growth doubles the texture each
/// time, so an unbounded atlas fed by a huge (or maliciously sized) glyph
/// would first exhaust memory and then overflow the GPU texture size math.
/// 4096 px is large enough for every realistic font size while keeping the
/// backing RGBA buffer at 64 MiB.
const MAX_ATLAS_SIZE: u16 = 4096;

impl Drop for Atlas {
    fn drop(&mut self) {
        let ctx = &mut get_context().quad_context;
        ctx.delete_texture(self.texture);
    }
}

impl Atlas {
    // pixel gap between glyphs in the atlas
    const GAP: u16 = 2;
    // well..
    const UNIQUENESS_OFFSET: u64 = 100000;

    pub fn new(ctx: &mut dyn miniquad::RenderingBackend, filter: miniquad::FilterMode) -> Atlas {
        let image = Image::gen_image_color(512, 512, Color::new(0.0, 0.0, 0.0, 0.0));
        let texture = ctx.new_texture_from_rgba8(image.width, image.height, &image.bytes);
        ctx.texture_set_filter(
            texture,
            miniquad::FilterMode::Nearest,
            miniquad::MipmapFilterMode::None,
        );

        Atlas {
            image,
            texture,
            cursor_x: 0,
            cursor_y: 0,
            dirty: false,
            max_line_height: 0,
            sprites: HashMap::new(),
            filter,
            unique_id: Self::UNIQUENESS_OFFSET,
        }
    }

    pub fn new_unique_id(&mut self) -> SpriteKey {
        self.unique_id += 1;

        SpriteKey::Id(self.unique_id)
    }

    pub fn set_filter(&mut self, filter_mode: miniquad::FilterMode) {
        let ctx = get_quad_context();
        self.filter = filter_mode;
        ctx.texture_set_filter(self.texture, filter_mode, miniquad::MipmapFilterMode::None);
    }

    pub fn get(&self, key: SpriteKey) -> Option<Sprite> {
        self.sprites.get(&key).cloned()
    }

    pub const fn width(&self) -> u16 {
        self.image.width
    }

    pub const fn height(&self) -> u16 {
        self.image.height
    }

    pub fn texture(&mut self) -> miniquad::TextureId {
        let ctx = get_quad_context();
        if self.dirty {
            self.dirty = false;
            let (texture_width, texture_height) = ctx.texture_size(self.texture);
            if texture_width != self.image.width as _ || texture_height != self.image.height as _ {
                ctx.delete_texture(self.texture);

                self.texture = ctx.new_texture_from_rgba8(
                    self.image.width,
                    self.image.height,
                    &self.image.bytes[..],
                );
                ctx.texture_set_filter(self.texture, self.filter, miniquad::MipmapFilterMode::None);
            }

            ctx.texture_update(self.texture, &self.image.bytes);
        }

        self.texture
    }

    pub fn get_uv_rect(&self, key: SpriteKey) -> Option<Rect> {
        let ctx = get_quad_context();
        self.get(key).map(|sprite| {
            let (w, h) = ctx.texture_size(self.texture);

            Rect::new(
                sprite.rect.x / w as f32,
                sprite.rect.y / h as f32,
                sprite.rect.w / w as f32,
                sprite.rect.h / h as f32,
            )
        })
    }

    pub fn cache_sprite(&mut self, key: SpriteKey, sprite: Image) {
        // A glyph wider or taller than the largest atlas we are willing to
        // allocate can never fit. Downscale the bitmap to the cap instead of
        // growing the atlas until the GPU size arithmetic overflows; the
        // sprite still occupies its (clamped) rect and renders as a blurry but
        // intact glyph rather than crashing the frame.
        let sprite = if sprite.width > MAX_ATLAS_SIZE || sprite.height > MAX_ATLAS_SIZE {
            downscale_image(sprite, MAX_ATLAS_SIZE)
        } else {
            sprite
        };

        let (width, height) = (sprite.width as usize, sprite.height as usize);

        let x = if self.cursor_x as u32 + width as u32 + Self::GAP as u32 <= self.image.width as u32
        {
            if height as u16 > self.max_line_height {
                self.max_line_height = height as u16;
            }
            let res = self.cursor_x + Self::GAP;
            self.cursor_x += width as u16 + Self::GAP * 2;
            res
        } else {
            self.cursor_y += self.max_line_height + Self::GAP * 2;
            self.cursor_x = width as u16 + Self::GAP;
            self.max_line_height = height as u16;
            Self::GAP
        };
        let y = self.cursor_y;

        // texture bounds exceeded
        if y + sprite.height > self.image.height || x + sprite.width > self.image.width {
            // The atlas has already reached its cap: there is nowhere left to
            // grow. Rather than overflowing the texture dimensions (or
            // recursing forever while the sprite still does not fit), evict
            // the whole cache once and place the incoming sprite in the fresh
            // buffer directly. Evicted glyphs are lazily re-rasterized later.
            // Evict when no amount of further doubling can fit the sprite:
            // one dimension is already at its cap while the sprite still
            // overflows it, or both axes have reached the cap. Growing past
            // this point would either never converge (infinite recursion) or
            // overflow the GPU texture size arithmetic.
            let growth_exhausted = (self.image.width >= MAX_ATLAS_SIZE
                && x + sprite.width > self.image.width)
                || (self.image.height >= MAX_ATLAS_SIZE
                    && y + sprite.height > self.image.height)
                || (self.image.width >= MAX_ATLAS_SIZE && self.image.height >= MAX_ATLAS_SIZE);
            if growth_exhausted {
                self.cursor_x = 0;
                self.cursor_y = 0;
                self.max_line_height = 0;
                self.sprites.clear();
                self.image = Image::gen_image_color(
                    MAX_ATLAS_SIZE,
                    MAX_ATLAS_SIZE,
                    Color::new(0.0, 0.0, 0.0, 0.0),
                );
                // The top-level downscale guarantees the sprite fits, but keep
                // a defensive clamp so this path can never re-enter the
                // bounds-exceeded branch and recurse indefinitely.
                let sprite = if sprite.width > MAX_ATLAS_SIZE || sprite.height > MAX_ATLAS_SIZE {
                    downscale_image(sprite, MAX_ATLAS_SIZE)
                } else {
                    sprite
                };
                // Fresh, empty buffer: place at the origin. A full-cap
                // sprite leaves no room for the usual GAP margin, whereas
                // anything smaller still fits at (0,0) regardless.
                self.max_line_height = sprite.height;
                self.cursor_x = sprite.width + Self::GAP;
                self.place_sprite(key, sprite, 0, 0);
                return;
            }

            // reset glyph cache state
            self.cursor_x = 0;
            self.cursor_y = 0;
            self.max_line_height = 0;

            let old_image = self.image.clone();

            // increase font texture size, clamped to the atlas cap.
            // note: if we tried to fit gigantic texture into a small atlas,
            // new_width will still be not enough. But its fine, it will
            // be regenerated on the recursion call.
            let new_width = (self.image.width * 2).min(MAX_ATLAS_SIZE);
            let new_height = (self.image.height * 2).min(MAX_ATLAS_SIZE);

            self.image =
                Image::gen_image_color(new_width, new_height, Color::new(0.0, 0.0, 0.0, 0.0));

            // recache all previously cached symbols
            // sprites are repacked tallest-first, ties broken by the previous
            // rect position: repacking in `HashMap` iteration order is
            // effectively random, which both wastes space on rows of mixed
            // heights and makes the atlas growth nondeterministic from run to
            // run. Ties must be broken deterministically as well — rect
            // positions are unique and themselves a product of the previous
            // deterministic packing, so this is a total order and the
            // key-to-rect mapping stays identical for every atlas instance.
            let mut sprites = self.sprites.drain().collect::<Vec<_>>();
            sprites.sort_by(|(_, a), (_, b)| {
                let key = |sprite: &Sprite| {
                    (
                        std::cmp::Reverse((sprite.rect.h as u32, sprite.rect.w as u32)),
                        sprite.rect.y as u32,
                        sprite.rect.x as u32,
                    )
                };
                key(a).cmp(&key(b))
            });
            for (key, sprite) in sprites {
                let image = old_image.sub_image(sprite.rect);
                self.cache_sprite(key, image);
            }

            // cache the new sprite
            self.cache_sprite(key, sprite);
        } else {
            self.place_sprite(key, sprite, x, y);
        }
    }

    /// Writes an already-positioned sprite's pixels at (`x`, `y`) and records
    /// its rect, advancing the row cursor. Callers must guarantee the sprite
    /// fits inside the current atlas image.
    fn place_sprite(&mut self, key: SpriteKey, sprite: Image, x: u16, y: u16) {
        let (width, height) = (sprite.width as usize, sprite.height as usize);

        debug_assert!(
            y + sprite.height <= self.image.height && x + sprite.width <= self.image.width,
            "place_sprite called with a sprite exceeding the atlas bounds"
        );

        self.dirty = true;
        for j in 0..height {
            for i in 0..width {
                self.image.set_pixel(
                    x as u32 + i as u32,
                    y as u32 + j as u32,
                    sprite.get_pixel(i as u32, j as u32),
                );
            }
        }

        self.sprites.insert(
            key,
            Sprite {
                rect: Rect::new(x as f32, y as f32, width as f32, height as f32),
            },
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn atlas(width: u16, height: u16) -> Atlas {
        let texture = miniquad::TextureId::from_raw_id(miniquad::RawId::OpenGl(0));

        Atlas {
            texture,
            image: Image::gen_image_color(width, height, Color::new(0.0, 0.0, 0.0, 0.0)),
            sprites: HashMap::new(),
            cursor_x: 0,
            cursor_y: 0,
            max_line_height: 0,
            dirty: false,
            filter: miniquad::FilterMode::Nearest,
            unique_id: Atlas::UNIQUENESS_OFFSET,
        }
    }

    fn sprite(width: u16, height: u16, color: Color) -> Image {
        Image::gen_image_color(width, height, color)
    }

    const WHITE: Color = Color::new(1.0, 1.0, 1.0, 1.0);

    // A sprite that would end past the right edge, but within one `GAP` of
    // it, used to take the "stays in the current row" branch and then blow
    // past the texture bounds, doubling the whole atlas instead of breaking
    // to the next row (issue #1054, bug 1).
    #[test]
    fn sprite_crossing_gap_at_row_edge_breaks_line_instead_of_growing() {
        let mut atlas = atlas(64, 1024);

        // 20 wide: placed at x = 2, cursor_x becomes 20 + 2 * GAP = 24
        atlas.cache_sprite(SpriteKey::Id(1), sprite(20, 10, WHITE));

        // 39 wide: cursor_x + width == 63, which passed the old
        // `cursor_x + width < image.width` check, yet the sprite lands at
        // x = 26 and ends at 65, one pixel past the 64-wide texture
        atlas.cache_sprite(SpriteKey::Id(2), sprite(39, 10, WHITE));

        // the sprite must have moved to the next row, not doubled the atlas
        assert_eq!(atlas.width(), 64);
        assert_eq!(atlas.height(), 1024);
        assert_eq!(
            atlas.get(SpriteKey::Id(2)).map(|sprite| sprite.rect),
            Some(Rect::new(2.0, 14.0, 39.0, 10.0))
        );

        std::mem::forget(atlas);
    }

    // The row-fit predicate must reject the out-of-bounds one-pixel case
    // above without wasting a row when a sprite ends exactly at the texture
    // edge. Its left-side GAP is already accounted for in `x`.
    #[test]
    fn sprite_ending_at_row_edge_stays_in_current_line() {
        let mut storage = std::mem::ManuallyDrop::new(atlas(64, 1024));
        let atlas = &mut *storage;

        atlas.cache_sprite(SpriteKey::Id(1), sprite(20, 10, WHITE));
        // x = 24 + GAP = 26, and x + width == 64 exactly.
        atlas.cache_sprite(SpriteKey::Id(2), sprite(38, 10, WHITE));

        assert_eq!(atlas.width(), 64);
        assert_eq!(atlas.height(), 1024);
        assert_eq!(
            atlas.get(SpriteKey::Id(2)).map(|sprite| sprite.rect),
            Some(Rect::new(26.0, 0.0, 38.0, 10.0))
        );
    }

    // Repacking the drained sprite cache in `HashMap` iteration order made
    // each regrow effectively shuffle the atlas, so the same glyph set could
    // grow to a different size on every run (issue #1054, bug 2). Sprites
    // are now repacked tallest-first and the resulting layout is exact.
    #[test]
    fn regrow_repacks_deterministically_and_keeps_pixels() {
        let mut atlas = atlas(64, 32);

        let red = Color::new(1.0, 0.0, 0.0, 1.0);
        let green = Color::new(0.0, 1.0, 0.0, 1.0);
        let blue = Color::new(0.0, 0.0, 1.0, 1.0);

        atlas.cache_sprite(SpriteKey::Id(1), sprite(20, 10, red));
        // ends within one `GAP` of the right edge on old code: forced a grow
        atlas.cache_sprite(SpriteKey::Id(2), sprite(39, 10, green));
        // 24 tall: does not fit the 32-tall atlas, forces exactly one grow
        atlas.cache_sprite(SpriteKey::Id(3), sprite(20, 24, blue));

        // the 64x32 atlas doubled once; tallest-first repacking puts the two
        // 10-tall sprites on row 0 (wider first) and the 24-tall one after
        // them, with no further growth
        assert_eq!((atlas.width(), atlas.height()), (128, 64));
        assert_eq!(
            atlas.get(SpriteKey::Id(1)).map(|sprite| sprite.rect),
            Some(Rect::new(45.0, 0.0, 20.0, 10.0))
        );
        assert_eq!(
            atlas.get(SpriteKey::Id(2)).map(|sprite| sprite.rect),
            Some(Rect::new(2.0, 0.0, 39.0, 10.0))
        );
        assert_eq!(
            atlas.get(SpriteKey::Id(3)).map(|sprite| sprite.rect),
            Some(Rect::new(69.0, 0.0, 20.0, 24.0))
        );

        // regrown sprites keep their pixels through the sub_image round-trip
        assert_eq!(atlas.image.get_pixel(45, 0), red);
        assert_eq!(atlas.image.get_pixel(2, 0), green);
        assert_eq!(atlas.image.get_pixel(69, 0), blue);

        std::mem::forget(atlas);
    }

    // Equal-size sprites must not inherit the randomized `HashMap` drain
    // order on regrow (issue #1054, bug 2): sorting by size alone leaves
    // same-dimension ties in arbitrary order, so which glyph ends up in
    // which slot — and the pixels under each key — differed between atlas
    // instances. Ties are now broken by the previous rect position, which is
    // unique and deterministic, so independently built atlases pack the
    // same-dimension sprites identically.
    #[test]
    fn same_dimension_sprites_repack_in_the_same_order_in_every_atlas() {
        let red = Color::new(1.0, 0.0, 0.0, 1.0);
        let green = Color::new(0.0, 1.0, 0.0, 1.0);
        let blue = Color::new(0.0, 0.0, 1.0, 1.0);
        let white = Color::new(1.0, 1.0, 1.0, 1.0);

        let pack = |atlas: &mut Atlas| {
            // two same-dimension sprites (20x10) next to each other
            atlas.cache_sprite(SpriteKey::Id(1), sprite(20, 10, red));
            atlas.cache_sprite(SpriteKey::Id(2), sprite(20, 10, green));
            // 24 tall: does not fit the 32-tall atlas, forces the first grow,
            // repacking the 20x10 tie
            atlas.cache_sprite(SpriteKey::Id(3), sprite(20, 24, blue));
            // 60 wide x 40 tall: fits neither the row nor the 64-tall atlas,
            // forcing a second grow whose repack hits the 20x10 tie again
            atlas.cache_sprite(SpriteKey::Id(4), sprite(60, 40, white));

            (
                (atlas.width(), atlas.height()),
                [
                    atlas.get(SpriteKey::Id(1)).map(|s| s.rect),
                    atlas.get(SpriteKey::Id(2)).map(|s| s.rect),
                    atlas.get(SpriteKey::Id(3)).map(|s| s.rect),
                    atlas.get(SpriteKey::Id(4)).map(|s| s.rect),
                ],
                [
                    atlas.image.get_pixel(2, 0),
                    atlas.image.get_pixel(26, 0),
                    atlas.image.get_pixel(50, 0),
                    atlas.image.get_pixel(74, 0),
                ],
            )
        };

        // independently built atlases must agree on the layout: every atlas
        // owns a separately seeded `HashMap`, so without a deterministic
        // tie-break the drained order of same-size sprites — and the layout
        // with it — would differ from atlas to atlas
        let mut first_atlas = atlas(64, 32);
        let first = pack(&mut first_atlas);
        for _ in 0..7 {
            let mut other = atlas(64, 32);
            assert_eq!(first, pack(&mut other));
            std::mem::forget(other);
        }

        // and the layout is the exact deterministic packing: two grows to
        // 256x128, tallest-first with rect-position tie-breaks — Id(3) and
        // the 20x10 pair ties always resolve by the previous position, so
        // Id(3) leads, then Id(1) before Id(2), and the 60x40 trigger is
        // appended after the repack without growing again
        let (size, rects, pixels) = first;
        assert_eq!(size, (256, 128));
        assert_eq!(
            rects,
            [
                Some(Rect::new(26.0, 0.0, 20.0, 10.0)),
                Some(Rect::new(50.0, 0.0, 20.0, 10.0)),
                Some(Rect::new(2.0, 0.0, 20.0, 24.0)),
                Some(Rect::new(74.0, 0.0, 60.0, 40.0)),
            ]
        );
        // each key still owns its own pixels after the double repack
        assert_eq!(pixels, [blue, red, green, white]);

        std::mem::forget(first_atlas);
    }

    // A glyph larger than the atlas cap must be downscaled and placed rather
    // than driving unbounded growth (which would overflow the GPU texture
    // size math) or recursing forever once the cap is reached.
    #[test]
    fn oversized_sprite_is_clamped_to_atlas_cap_without_recursion() {
        let mut storage = std::mem::ManuallyDrop::new(atlas(64, 64));
        let atlas = &mut *storage;

        // Seed a normal sprite first so the eviction path has something to
        // clear.
        atlas.cache_sprite(SpriteKey::Id(1), sprite(10, 10, WHITE));

        // A sprite well above the 4096px cap would otherwise grow the atlas
        // until allocation or arithmetic fails.
        let huge = sprite(super::MAX_ATLAS_SIZE + 1500, 20, red());
        atlas.cache_sprite(SpriteKey::Id(2), huge);

        let rect = atlas.get(SpriteKey::Id(2)).expect("huge sprite is stored").rect;
        assert!(rect.w <= super::MAX_ATLAS_SIZE as f32);
        assert!(rect.h <= super::MAX_ATLAS_SIZE as f32);
        assert!(rect.w >= 1.0 && rect.h >= 1.0);
        // The atlas never grows beyond its hard cap.
        assert!(atlas.width() <= super::MAX_ATLAS_SIZE);
        assert!(atlas.height() <= super::MAX_ATLAS_SIZE);

        // A second oversized sprite reuses the eviction path and still
        // terminates (the original bug was infinite recursion here).
        let huge2 = sprite(20, super::MAX_ATLAS_SIZE + 2000, green());
        atlas.cache_sprite(SpriteKey::Id(3), huge2);
        assert!(atlas.get(SpriteKey::Id(3)).is_some());
    }

    fn red() -> Color {
        Color::new(1.0, 0.0, 0.0, 1.0)
    }

    fn green() -> Color {
        Color::new(0.0, 1.0, 0.0, 1.0)
    }
}
