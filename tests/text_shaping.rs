//! End-to-end tests for the complex text shaping pipeline.
//!
//! The pure shaping logic (joining tables, matra reordering, bidi reversal)
//! is covered by unit tests in `src/text/shaping.rs`; this file exercises the
//! full path through font coverage resolution, glyph rasterization, the
//! shaping cache and the draw call itself.
//!
//! miniquad owns a process-global display, so only one `Window` can exist per
//! test process. All scenarios therefore live in a single
//! `#[macroquad::test]` (the same constraint the existing macroquad test
//! suite follows); run the file with `cargo test --test text_shaping`.

use macroquad::prelude::*;

#[macroquad::test]
async fn complex_text_shaping_end_to_end() {
    // --- Arabic: contextual forms, ligatures and RTL drawing -------------
    let arabic = "العربية";
    let shaped = measure_text_shaped(arabic, None, 30, 1.0);
    assert!(shaped.width > 0.0);
    let legacy = measure_text(arabic, None, 30, 1.0);
    assert!(shaped.width >= legacy.width * 0.5);

    let dims = draw_text_ex(
        arabic,
        20.0,
        40.0,
        TextParams {
            font_size: 30,
            complex_text: true,
            ..Default::default()
        },
    );
    assert!(dims.width > 0.0);
    // Repeating the same string must hit the shaping cache: identical width.
    let dims_cached = draw_text_ex(
        arabic,
        20.0,
        80.0,
        TextParams {
            font_size: 30,
            complex_text: true,
            ..Default::default()
        },
    );
    assert!((dims.width - dims_cached.width).abs() < f32::EPSILON);

    // --- Devanagari: short-i matra reordering and virama clusters --------
    let deva = "किताब";
    let dims = draw_text_ex(
        deva,
        20.0,
        120.0,
        TextParams {
            font_size: 28,
            complex_text: true,
            ..Default::default()
        },
    );
    assert!(dims.width > 0.0);
    let conjunct = "क्ति";
    let dims = draw_text_ex(
        conjunct,
        20.0,
        160.0,
        TextParams {
            font_size: 28,
            complex_text: true,
            ..Default::default()
        },
    );
    assert!(dims.width > 0.0);

    // --- Emoji / missing glyphs: replacement glyph, no crash -------------
    // The bundled ProggyClean font covers none of the emoji codepoints.
    let mixed = "a😀b🎉c";
    let dims = draw_text_ex(
        mixed,
        10.0,
        200.0,
        TextParams {
            font_size: 24,
            complex_text: true,
            ..Default::default()
        },
    );
    let latin_width = measure_text("abc", None, 24, 1.0).width;
    assert!(dims.width >= latin_width);
    let dims = draw_text_lossy("a\u{FFFD}b".as_bytes(), 10.0, 230.0, 24.0, WHITE);
    assert!(dims.width > 0.0);

    // --- Invalid UTF-8: replacement instead of panic ---------------------
    let bytes = b"ab\xFF\xFEcd";
    let dims = draw_text_lossy(bytes, 10.0, 260.0, 20.0, WHITE);
    assert!(dims.width > measure_text("abcd", None, 20, 1.0).width);
    let dims = draw_text_lossy(&[0xE0, 0xA4], 10.0, 290.0, 20.0, WHITE);
    assert!(dims.width > 0.0);

    // --- Newlines: all three conventions split into stacked lines --------
    let multiline = "line1\nline2\r\nline3\rline4";
    let dims = draw_text_ex(
        multiline,
        10.0,
        330.0,
        TextParams {
            font_size: 20,
            complex_text: true,
            ..Default::default()
        },
    );
    let one_line = measure_text("line1", None, 20, 1.0);
    assert!(dims.height >= one_line.height * 3.0);
    assert!(dims.width > 0.0);

    // --- Zero-width characters: invisible, zero width --------------------
    let plain_w = measure_text_shaped("abc", None, 20, 1.0).width;
    let decorated_w =
        measure_text_shaped("a\u{200B}b\u{200D}c\u{200C}\u{FEFF}", None, 20, 1.0).width;
    assert!((plain_w - decorated_w).abs() < 0.01);

    // --- Oversized font size: clamped, no atlas OOM ----------------------
    // Use a dedicated font so its atlas growth cannot evict sprites cached
    // for the shared default font used by the measurements above.
    let huge_font =
        load_ttf_font_from_bytes(include_bytes!("../examples/DancingScriptRegular.ttf"))
            .expect("bundled TTF must load");
    let dims = draw_text_ex(
        "abc",
        0.0,
        380.0,
        TextParams {
            font: Some(&huge_font),
            font_size: u16::MAX,
            complex_text: true,
            ..Default::default()
        },
    );
    assert!(dims.width > 0.0 && dims.height > 0.0);

    // --- Fallback chain: registration invalidates cache safely -----------
    // A separately loaded font instance keeps its own atlas and character
    // map, so adding it to the chain cannot collide with the primary font's
    // sprite keys.
    let font = get_default_font();
    let fallback = load_ttf_font_from_bytes(include_bytes!("../examples/DancingScriptRegular.ttf"))
        .expect("bundled TTF must load");
    font.add_fallback_font(fallback);
    let dims = draw_text_ex(
        "مرحبا 123",
        10.0,
        420.0,
        TextParams {
            font: Some(&font),
            font_size: 26,
            complex_text: true,
            ..Default::default()
        },
    );
    assert!(dims.width > 0.0);
    font.add_fallback_font(
        load_ttf_font_from_bytes(include_bytes!("../examples/DancingScriptRegular.ttf"))
            .expect("bundled TTF must load"),
    );
    let dims2 = draw_text_ex(
        "مرحبا 123",
        10.0,
        460.0,
        TextParams {
            font: Some(&font),
            font_size: 26,
            complex_text: true,
            ..Default::default()
        },
    );
    assert!((dims.width - dims2.width).abs() < f32::EPSILON);

    // --- Legacy path unchanged when complex_text is false ----------------
    let text = "Legacy macroquad text 0123";
    let legacy_draw = draw_text_ex(
        text,
        5.0,
        500.0,
        TextParams {
            font_size: 22,
            complex_text: false,
            ..Default::default()
        },
    );
    let measured = measure_text(text, None, 22, 1.0);
    assert!((legacy_draw.width - measured.width).abs() < 0.01);
    assert!((legacy_draw.height - measured.height).abs() < 0.01);

    // --- Latin width parity between the two pipelines --------------------
    // For a script the default font fully covers, the shaped pipeline must
    // report the same advance width as the legacy codepoint pipeline (same
    // glyphs, same advances, kerning applied in both), which is the
    // measurable guarantee behind "legacy output stays unchanged".
    for latin in ["macroquad", "The quick brown fox", "42.0 px"] {
        let legacy_w = measure_text(latin, None, 24, 1.0).width;
        let shaped_w = measure_text_shaped(latin, None, 24, 1.0).width;
        assert!(
            (legacy_w - shaped_w).abs() < 0.05,
            "width mismatch for {latin:?}: legacy={legacy_w} shaped={shaped_w}"
        );
    }
}
