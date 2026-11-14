//! Complex text shaping example.
//!
//! Demonstrates the opt-in `complex_text` pipeline: Arabic contextual forms
//! and RTL ordering, Devanagari matra reordering, combining marks, font
//! fallback, emoji/missing-glyph replacement and multiline layout, all
//! rendered alongside the legacy codepoint-by-codepoint path for comparison.

use macroquad::prelude::*;

#[macroquad::main("Text shaping")]
async fn main() {
    // A second font registered as a fallback. ProggyClean (the default) only
    // covers Latin, so Arabic/Devanagari codepoints resolve through the
    // fallback chain; when no font in the chain covers a codepoint the
    // replacement glyph is rendered. DejaVu Sans ships on most Linux desktops
    // and covers both Arabic and Devanagari.
    let font = get_default_font();
    if let Ok(bytes) = load_file("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf").await {
        if let Ok(cover_font) = load_ttf_font_from_bytes(&bytes) {
            font.add_fallback_font(cover_font);
        }
    }
    // The bundled DancingScript stays in the chain as a Latin display face,
    // demonstrating multi-font fallback ordering.
    if let Ok(dancing) = load_ttf_font("./examples/DancingScriptRegular.ttf").await {
        font.add_fallback_font(dancing);
    }

    let samples: &[(&str, &str)] = &[
        ("Arabic (RTL, joining):", "العربية"),
        ("Arabic lam-alef ligature:", "لا"),
        ("Devanagari (matra reorder):", "किताब"),
        ("Devanagari conjunct:", "क्ति"),
        ("Combining marks:", "a\u{0301}e\u{0302}i\u{0308}"),
        ("Emoji -> fallback glyph:", "hello 😀 world"),
        ("Zero-width chars:", "a\u{200B}b\u{200D}c"),
        ("Mixed scripts + digits:", "abc ديج 123"),
        ("Two lines:", "first line\nsecond line"),
    ];

    loop {
        clear_background(BLACK);

        draw_text("Complex text shaping (complex_text: true)", 20.0, 30.0, 24.0, WHITE);
        draw_text(
            "Same strings through the legacy path (complex_text: false)",
            430.0,
            30.0,
            20.0,
            GRAY,
        );

        let mut y = 70.0_f32;
        for (label, sample) in samples {
            draw_text(label, 20.0, y, 16.0, SKYBLUE);

            draw_text_ex(
                sample,
                20.0,
                y + 28.0,
                TextParams {
                    font: Some(&font),
                    font_size: 26,
                    complex_text: true,
                    ..Default::default()
                },
            );

            // Legacy rendering of the identical string, side by side.
            draw_text_ex(
                sample,
                430.0,
                y + 28.0,
                TextParams {
                    font: Some(&font),
                    font_size: 26,
                    complex_text: false,
                    ..Default::default()
                },
            );

            y += 70.0;
        }

        // Invalid UTF-8 never reaches draw_text via &str; the lossy entry
        // point replaces bad bytes with U+FFFD instead.
        let bad_bytes: &[u8] = b"valid \xFF\xFE end";
        let dims = draw_text_lossy(bad_bytes, 20.0, screen_height() - 30.0, 22.0, RED);
        draw_text(
            format!("lossy decode width: {:.1}px", dims.width).as_str(),
            320.0,
            screen_height() - 30.0,
            18.0,
            GREEN,
        );

        next_frame().await;
    }
}
