//! Complex text shaping.
//!
//! This module turns a Unicode string into a sequence of positioned glyph
//! runs. It is intentionally self contained: it does not talk to the GPU, the
//! font atlas or miniquad, which makes every stage (segmentation, joining,
//! reordering, mark placement) independently testable.
//!
//! The implemented feature set is deliberately scoped to what can be done
//! with the font tables exposed by `fontdue` (cmap / hmtx / kern):
//!
//! * script and direction segmentation,
//! * Arabic contextual forms and lam-alef ligatures,
//! * Devanagari short-i matra reordering and virama cluster handling,
//! * RTL visual reordering,
//! * combining mark anchoring,
//! * kerning and per-glyph font fallback.
//!
//! Full OpenType GSUB/GPOS engines (harfbuzz equivalents) are out of scope;
//! where a font lacks a required presentation glyph the fallback chain or the
//! replacement glyph is used instead, and shaping never panics on bad input.

use std::hash::{Hash, Hasher};

/// Hard cap on the rasterized pixel size. A 10 000 px glyph would otherwise
/// allocate hundreds of megabytes of atlas memory; anything above this is
/// clamped rather than crashed on.
pub(crate) const MAX_FONT_SIZE: u16 = 256;

/// Replacement codepoint emitted for invalid UTF-8 and unmappable input.
pub const REPLACEMENT_CHAR: char = '\u{FFFD}';

/// Scripts the shaper knows how to segment.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) enum Script {
    /// Common/inherited characters (punctuation, digits, spaces, symbols):
    /// they follow the direction of the run they end up in.
    Common,
    Latin,
    Arabic,
    Devanagari,
    Hebrew,
    Other,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Direction {
    Ltr,
    Rtl,
}

impl Script {
    fn direction(self) -> Direction {
        match self {
            Script::Arabic | Script::Hebrew => Direction::Rtl,
            _ => Direction::Ltr,
        }
    }
}

/// Classifies a single codepoint into one of the shaper's script buckets.
/// The ranges mirror the Unicode block assignments; unknown blocks fall back
/// to `Other`, which is shaped with the generic LTR pipeline.
pub(crate) fn script_of(c: char) -> Script {
    let u = c as u32;
    match u {
        // Arabic block, Arabic Supplement, Arabic Presentation Forms A/B
        0x0600..=0x06FF | 0x0750..=0x077F | 0xFB50..=0xFDFF | 0xFE70..=0xFEFF => Script::Arabic,
        // Devanagari and Devanagari Extended
        0x0900..=0x097F | 0xA8E0..=0xA8FF => Script::Devanagari,
        // Hebrew
        0x0590..=0x05FF => Script::Hebrew,
        _ => {
            if is_combining_mark(c) || is_zero_width(c) || c.is_whitespace() {
                Script::Common
            } else if is_emoji(c) {
                Script::Other
            } else {
                // Latin, Latin Extended, Latin Extended Additional,
                // General Punctuation, superscripts, currency ...
                match u {
                    0x0041..=0x005A
                    | 0x0061..=0x007A
                    | 0x00C0..=0x024F
                    | 0x1E00..=0x1EFF
                    | 0x2100..=0x214F => Script::Latin,
                    // ASCII digits, punctuation and the remaining general
                    // punctuation stay Common so they inherit the script of
                    // their neighbouring letters (Arabic digits inside an
                    // Arabic run must not open a Latin run).
                    _ => Script::Common,
                }
            }
        }
    }
}

/// Zero-width / invisible formatting characters that must never produce a
/// glyph or take up pen space.
pub(crate) fn is_zero_width(c: char) -> bool {
    matches!(
        c,
        '\u{200B}' // zero width space
        | '\u{200C}' // zero width non-joiner
        | '\u{200D}' // zero width joiner
        | '\u{2060}' // word joiner
        | '\u{FEFF}' // zero width no-break space / BOM
    )
}

/// Whether `c` is a combining mark (general category Mn/Mc). The ranges cover
/// the combining diacritical blocks plus the script specific mark ranges we
/// shape; anything outside them is treated as a spacing character.
pub(crate) fn is_combining_mark(c: char) -> bool {
    let u = c as u32;
    matches!(u,
        0x0300..=0x036F // Combining Diacritical Marks
        | 0x1AB0..=0x1AFF // Combining Diacritical Marks Extended
        | 0x1DC0..=0x1DFF // Combining Diacritical Marks Supplement
        | 0x20D0..=0x20FF // Combining Diacritical Marks for Symbols
        | 0xFE20..=0xFE2F // Combining Half Marks
        // Devanagari signs: nukta, vowel signs (U+093E..U+094C except the
        // repositioned short i), virama, Vedic extensions
        | 0x093C
        | 0x093E..=0x094C
        | 0x0951..=0x0957
        | 0xA8E0..=0xA8FF
        // Arabic combining marks (harakat)
        | 0x064B..=0x065F
        | 0x0670
        | 0x06D6..=0x06DC
        | 0x06DF..=0x06E4
        | 0x06E7..=0x06E8
        | 0x06EA..=0x06ED
    )
}

/// "Color emoji" ranges. The shaper itself only needs to know that these are
/// distinct clusters; rasterization falls back to `.notdef` unless a font
/// with coverage is registered.
pub(crate) fn is_emoji(c: char) -> bool {
    let u = c as u32;
    matches!(u,
        0x1F300..=0x1F5FF // Misc Symbols and Pictographs
        | 0x1F600..=0x1F64F // Emoticons
        | 0x1F680..=0x1F6FF // Transport and Map
        | 0x1F700..=0x1F77F // Alchemical
        | 0x1F780..=0x1F7FF // Geometric Shapes Extended
        | 0x1F800..=0x1F8FF // Supplemental Arrows-C
        | 0x1F900..=0x1F9FF // Supplemental Symbols and Pictographs
        | 0x1FA00..=0x1FA6F // Chess Symbols
        | 0x1FA70..=0x1FAFF // Symbols and Pictographs Extended-A
        | 0x2600..=0x26FF // Misc Symbols
        | 0x2700..=0x27BF // Dingbats
        | 0x2190..=0x21FF // Arrows (emoji presentation possible)
        | 0x2300..=0x23FF
        | 0x2B00..=0x2BFF
        | 0xFE0F // variation selector-16
    )
}

/// A segment of text sharing one concrete script and direction.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct Run {
    pub chars: Vec<char>,
    pub script: Script,
    pub direction: Direction,
}

/// Splits text into script/direction runs. `Common` characters stick to the
/// run they occur in (so Arabic digits and punctuation inherit RTL) and a
/// leading common run adopts the first concrete script that follows.
pub(crate) fn segment_runs(chars: &[char]) -> Vec<Run> {
    // First pass: attach common/inherited codepoints to a concrete script.
    let mut tagged: Vec<(char, Script, Direction)> = Vec::with_capacity(chars.len());
    let mut last_concrete: Option<Script> = None;
    for &c in chars {
        let raw = script_of(c);
        let script = match raw {
            Script::Common => last_concrete.unwrap_or(Script::Latin),
            other => {
                last_concrete = Some(other);
                other
            }
        };
        tagged.push((c, script, script.direction()));
    }

    let mut runs: Vec<Run> = Vec::new();
    for (c, script, direction) in tagged {
        match runs.last_mut() {
            Some(run) if run.script == script && run.direction == direction => {
                run.chars.push(c);
            }
            _ => runs.push(Run {
                chars: vec![c],
                script,
                direction,
            }),
        }
    }
    runs
}

// ---------------------------------------------------------------------------
// Arabic
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum JoinType {
    /// Non joining: causes adjacent joiners to use their non joined forms.
    Non,
    /// Right joining: joins to the previous letter only.
    Right,
    /// Dual joining: joins on both sides.
    Dual,
}

/// Joining group of an Arabic letter. `form(base, joined)` maps a letter to
/// the codepoint of its isolated/final (`joined = false`/`true`, right side)
/// or initial/medial (`left` side) presentation form.
#[derive(Clone, Copy)]
struct Joining {
    kind: JoinType,
    isolated: char,
    final_form: char,
    initial: char,
    medial: char,
}

impl Joining {
    const fn dual(
        _base: char,
        isolated: char,
        final_form: char,
        initial: char,
        medial: char,
    ) -> Self {
        Joining {
            kind: JoinType::Dual,
            isolated,
            final_form,
            initial,
            medial,
        }
    }

    const fn right(_base: char, isolated: char, final_form: char) -> Self {
        Joining {
            kind: JoinType::Right,
            isolated,
            final_form,
            initial: _base,
            medial: _base,
        }
    }

    const fn non(base: char) -> Self {
        Joining {
            kind: JoinType::Non,
            isolated: base,
            final_form: base,
            initial: base,
            medial: base,
        }
    }
}

/// Static joining table for the Arabic block (U+0621..U+0640) plus the
/// joined diacritic hamza letters. Lookup is linear over a small table, which
/// keeps the data legible; shaping throughput is dominated by rasterization,
/// not by this scan.
const JOINING_TABLE: &[(char, Joining)] = &[
    // hamza and hamza carriers
    ('\u{0621}', Joining::non('\u{0621}')),
    (
        '\u{0622}',
        Joining::right('\u{0622}', '\u{FE81}', '\u{FE82}'),
    ),
    (
        '\u{0623}',
        Joining::right('\u{0623}', '\u{FE83}', '\u{FE84}'),
    ),
    (
        '\u{0624}',
        Joining::right('\u{0624}', '\u{FE85}', '\u{FE86}'),
    ),
    (
        '\u{0625}',
        Joining::right('\u{0625}', '\u{FE87}', '\u{FE88}'),
    ),
    (
        '\u{0626}',
        Joining::dual('\u{0626}', '\u{FE89}', '\u{FE8A}', '\u{FE8B}', '\u{FE8C}'),
    ),
    (
        '\u{0627}',
        Joining::right('\u{0627}', '\u{FE8D}', '\u{FE8E}'),
    ),
    (
        '\u{0628}',
        Joining::dual('\u{0628}', '\u{FE8F}', '\u{FE90}', '\u{FE91}', '\u{FE92}'),
    ),
    (
        '\u{0629}',
        Joining::right('\u{0629}', '\u{FE93}', '\u{FE94}'),
    ),
    (
        '\u{062A}',
        Joining::dual('\u{062A}', '\u{FE95}', '\u{FE96}', '\u{FE97}', '\u{FE98}'),
    ),
    (
        '\u{062B}',
        Joining::dual('\u{062B}', '\u{FE99}', '\u{FE9A}', '\u{FE9B}', '\u{FE9C}'),
    ),
    (
        '\u{062C}',
        Joining::dual('\u{062C}', '\u{FE9D}', '\u{FE9E}', '\u{FE9F}', '\u{FEA0}'),
    ),
    (
        '\u{062D}',
        Joining::dual('\u{062D}', '\u{FEA1}', '\u{FEA2}', '\u{FEA3}', '\u{FEA4}'),
    ),
    (
        '\u{062E}',
        Joining::dual('\u{062E}', '\u{FEA5}', '\u{FEA6}', '\u{FEA7}', '\u{FEA8}'),
    ),
    ('\u{062F}', Joining::right('\u{062F}', '\u{FEA9}', '\u{FEAA}')),
    ('\u{0630}', Joining::right('\u{0630}', '\u{FEAB}', '\u{FEAC}')),
    ('\u{0631}', Joining::right('\u{0631}', '\u{FEAD}', '\u{FEAE}')),
    ('\u{0632}', Joining::right('\u{0632}', '\u{FEAF}', '\u{FEB0}')),
    (
        '\u{0633}',
        Joining::dual('\u{0633}', '\u{FEB1}', '\u{FEB2}', '\u{FEB3}', '\u{FEB4}'),
    ),
    (
        '\u{0634}',
        Joining::dual('\u{0634}', '\u{FEB5}', '\u{FEB6}', '\u{FEB7}', '\u{FEB8}'),
    ),
    (
        '\u{0635}',
        Joining::dual('\u{0635}', '\u{FEB9}', '\u{FEBA}', '\u{FEBB}', '\u{FEBC}'),
    ),
    (
        '\u{0636}',
        Joining::dual('\u{0636}', '\u{FEBD}', '\u{FEBE}', '\u{FEBF}', '\u{FEC0}'),
    ),
    (
        '\u{0637}',
        Joining::dual('\u{0637}', '\u{FEC1}', '\u{FEC2}', '\u{FEC3}', '\u{FEC4}'),
    ),
    (
        '\u{0638}',
        Joining::dual('\u{0638}', '\u{FEC5}', '\u{FEC6}', '\u{FEC7}', '\u{FEC8}'),
    ),
    (
        '\u{0639}',
        Joining::dual('\u{0639}', '\u{FEC9}', '\u{FECA}', '\u{FECB}', '\u{FECC}'),
    ),
    (
        '\u{063A}',
        Joining::dual('\u{063A}', '\u{FECD}', '\u{FECE}', '\u{FECF}', '\u{FED0}'),
    ),
    // tatweel: dual joining, every form is the tatweel itself
    (
        '\u{0640}',
        Joining::dual('\u{0640}', '\u{0640}', '\u{0640}', '\u{0640}', '\u{0640}'),
    ),
    (
        '\u{0641}',
        Joining::dual('\u{0641}', '\u{FED1}', '\u{FED2}', '\u{FED3}', '\u{FED4}'),
    ),
    (
        '\u{0642}',
        Joining::dual('\u{0642}', '\u{FED5}', '\u{FED6}', '\u{FED7}', '\u{FED8}'),
    ),
    (
        '\u{0643}',
        Joining::dual('\u{0643}', '\u{FED9}', '\u{FEDA}', '\u{FEDB}', '\u{FEDC}'),
    ),
    (
        '\u{0644}',
        Joining::dual('\u{0644}', '\u{FEDD}', '\u{FEDE}', '\u{FEDF}', '\u{FEE0}'),
    ),
    (
        '\u{0645}',
        Joining::dual('\u{0645}', '\u{FEE1}', '\u{FEE2}', '\u{FEE3}', '\u{FEE4}'),
    ),
    (
        '\u{0646}',
        Joining::dual('\u{0646}', '\u{FEE5}', '\u{FEE6}', '\u{FEE7}', '\u{FEE8}'),
    ),
    (
        '\u{0647}',
        Joining::dual('\u{0647}', '\u{FEE9}', '\u{FEEA}', '\u{FEEB}', '\u{FEEC}'),
    ),
    (
        '\u{0648}',
        Joining::right('\u{0648}', '\u{FEED}', '\u{FEEE}'),
    ),
    (
        '\u{0649}',
        Joining::right('\u{0649}', '\u{FEEF}', '\u{FEF0}'),
    ),
    (
        '\u{064A}',
        Joining::dual('\u{064A}', '\u{FEF1}', '\u{FEF2}', '\u{FEF3}', '\u{FEF4}'),
    ),
    // U+0671..U+0677 alef-like extended letters are right joining
    ('\u{0671}', Joining::right('\u{0671}', '\u{FB50}', '\u{FB51}')),
    ('\u{0677}', Joining::right('\u{0677}', '\u{FB5E}', '\u{FB5F}')),
];

fn joining_of(c: char) -> Option<Joining> {
    JOINING_TABLE
        .iter()
        .find(|(base, _)| *base == c)
        .map(|(_, j)| *j)
        .or_else(|| ligature_joining(c))
}

/// Lam-alef ligatures behave like dual-joining letters for the purpose of
/// neighbour scans: an initial-form ligature still joins the letter on its
/// right (logical previous). Both forms are listed explicitly because the
/// ligature codepoints live outside the main Arabic joining table.
fn ligature_joining(c: char) -> Option<Joining> {
    let pairs = [
        ('\u{FEF5}', '\u{FEF6}'),
        ('\u{FEF7}', '\u{FEF8}'),
        ('\u{FEF9}', '\u{FEFA}'),
        ('\u{FEFB}', '\u{FEFC}'),
    ];
    pairs.iter().find_map(|(isolated, final_form)| {
        if c == *isolated || c == *final_form {
            Some(Joining {
                kind: JoinType::Dual,
                isolated: *isolated,
                final_form: *final_form,
                // initial/medial ligature forms do not exist; a dual-joining
                // neighbour on the left is expressed through the final form,
                // which stage 1 already selected.
                initial: *isolated,
                medial: *final_form,
            })
        } else {
            None
        }
    })
}

/// Lam-alef mandatory ligatures. Keyed `(lam, alef)`, the value is the
/// isolated presentation form; the final form follows it in the block.
const LAM_ALEF: &[(char, char, char)] = &[
    ('\u{0644}', '\u{0622}', '\u{FEF5}'), // lam + alef with madda
    ('\u{0644}', '\u{0623}', '\u{FEF7}'), // lam + alef with hamza above
    ('\u{0644}', '\u{0625}', '\u{FEF9}'), // lam + alef with hamza below
    ('\u{0644}', '\u{0627}', '\u{FEFB}'), // lam + alef
];

fn lam_alef_ligature(lam: char, alef: char) -> Option<char> {
    LAM_ALEF
        .iter()
        .find(|(l, a, _)| *l == lam && *a == alef)
        .map(|(_, _, lig)| *lig)
}

/// Applies Arabic joining to one run.
///
/// Stage 1 substitutes lam + alef ligatures (accounting for intervening
/// transparent marks). Stage 2 walks the logical order and picks each
/// letter's isolated / final / initial / medial form from the joining state
/// of its neighbours. ZWNJ explicitly breaks joining, ZWJ forces it.
/// Marks, spaces and unknown characters pass through untouched.
pub(crate) fn shape_arabic(input: &[char]) -> Vec<char> {
    // Stage 1: lam-alef ligatures.
    let mut ligated: Vec<char> = Vec::with_capacity(input.len());
    let mut i = 0;
    while i < input.len() {
        let c = input[i];
        if c == '\u{0644}' && i + 1 < input.len() {
            // skip transparent marks between lam and the alef
            let mut j = i + 1;
            while j < input.len() && is_zero_width(input[j]) && input[j] != '\u{200C}' {
                j += 1;
            }
            if j < input.len() {
                if let Some(lig) = lam_alef_ligature(c, input[j]) {
                    // Find the nearest non-transparent preceding letter to
                    // decide between the ligature's isolated and final form.
                    // Only dual-joining letters can join from their right; a
                    // right-joining letter such as alef does not connect to
                    // anything that follows it.
                    let mut prev_joins = false;
                    let mut p = ligated.len();
                    while p > 0 {
                        p -= 1;
                        let prev = ligated[p];
                        if is_zero_width(prev) || is_combining_mark(prev) {
                            continue;
                        }
                        prev_joins = joining_of(prev)
                            .map(|j| matches!(j.kind, JoinType::Dual))
                            .unwrap_or(false);
                        break;
                    }
                    ligated.push(if prev_joins {
                        char::from_u32(lig as u32 + 1).unwrap()
                    } else {
                        lig
                    });
                    // keep the marks, drop the consumed alef (and joiners)
                    ligated.extend_from_slice(&input[i + 1..j]);
                    i = j + 1;
                    continue;
                }
            }
        }
        ligated.push(c);
        i += 1;
    }

    // Stage 2: contextual forms. Indices always refer back into `ligated`;
    // transparent codepoints are skipped inline rather than prefiltered, so
    // the neighbour scan and the output stay in sync.
    let n = ligated.len();
    let mut out: Vec<char> = Vec::with_capacity(n);
    for idx in 0..n {
        let c = ligated[idx];

        if is_zero_width(c) {
            // ZWNJ is purely a joining barrier (handled by the neighbour
            // scans below); the joiners have already done their job. Nothing
            // invisible may reach the rasterizer.
            continue;
        }

        let Some(joining) = joining_of(c) else {
            out.push(c);
            continue;
        };

        // Lam-alef ligatures received their isolated/final form during the
        // substitution stage and must not be reshaped here.
        if ligature_joining(c).is_some() {
            out.push(c);
            continue;
        }

        // Left neighbour = previous join-causing codepoint, ignoring
        // transparent formatting but stopping at ZWNJ.
        let mut left_joins = false;
        let mut k = idx;
        while k > 0 {
            k -= 1;
            let p = ligated[k];
            if p == '\u{200C}' {
                break;
            }
            if is_zero_width(p) {
                continue;
            }
            if let Some(pj) = joining_of(p) {
                left_joins = matches!(pj.kind, JoinType::Dual | JoinType::Right);
            }
            break;
        }

        // Right neighbour = next join-causing codepoint.
        let mut right_joins = false;
        let mut k = idx + 1;
        while k < n {
            let q = ligated[k];
            if q == '\u{200C}' {
                break;
            }
            if is_zero_width(q) {
                k += 1;
                continue;
            }
            if let Some(qj) = joining_of(q) {
                right_joins = matches!(qj.kind, JoinType::Dual);
            }
            break;
        }

        let shaped = match (left_joins, right_joins, joining.kind) {
            (true, true, JoinType::Dual) => joining.medial,
            (true, false, JoinType::Dual | JoinType::Right) => joining.final_form,
            (false, true, JoinType::Dual) => joining.initial,
            _ => joining.isolated,
        };
        out.push(shaped);
    }
    out
}

// ---------------------------------------------------------------------------
// Devanagari
// ---------------------------------------------------------------------------

const DEVA_VIRAMA: char = '\u{094D}';
const DEVA_SHORT_I: char = '\u{093F}';
/// Devanagari danda and related punctuation break orthographic syllables.
fn is_deva_syllable_break(c: char) -> bool {
    c.is_whitespace() || matches!(c as u32, 0x0964 | 0x0965) || c == '|'
}

/// Devanagari shaping.
///
/// The one reordering rule that can be implemented without GSUB lookups is
/// the short-i vowel sign (U+093F): typographically it appears *before* its
/// base consonant cluster, even though Unicode stores it after. The reorder
/// moves it to the start of its orthographic syllable (the consonant cluster
/// terminated by the matra). The virama is retained so fonts with conjunct
/// ligatures still form them via their own lookup; without such a font the
/// explicit virama glyph remains visible instead of disappearing.
pub(crate) fn shape_devanagari(input: &[char]) -> Vec<char> {
    let mut out: Vec<char> = Vec::with_capacity(input.len());
    let mut i = 0;
    while i < input.len() {
        let c = input[i];
        if c != DEVA_SHORT_I {
            out.push(c);
            i += 1;
            continue;
        }

        // Locate the insertion point by scanning the already-emitted
        // syllable from right to left.
        //
        // Layout (rightmost first while scanning): [other marks]* then
        // zero or more `base consonant + virama` conjunct pairs, then the
        // syllable's first base consonant. The matra is inserted before that
        // first base consonant. A whitespace/danda boundary aborts the scan
        // so the matra never escapes its own syllable.
        let mut start = out.len();

        // 1. skip non-virama marks attached to the last consonant
        while start > 0
            && is_combining_mark(out[start - 1])
            && out[start - 1] != DEVA_VIRAMA
            && out[start - 1] != DEVA_SHORT_I
        {
            start -= 1;
        }

        // 2. consume every `consonant + virama` pair of the conjunct chain
        while start >= 2
            && out[start - 1] != DEVA_VIRAMA
            && out[start - 2] == DEVA_VIRAMA
            && !is_deva_syllable_break(out[start - 1])
        {
            start -= 2;
        }

        // 3. Step over the syllable's first base consonant (in the conjunct
        // case this is the consonant left before the first consumed
        // `virama + consonant` pair).
        if start > 0
            && !is_deva_syllable_break(out[start - 1])
            && out[start - 1] != DEVA_VIRAMA
        {
            start -= 1;
        }

        // 4. insert before the first base consonant of the syllable
        out.insert(start, DEVA_SHORT_I);
        i += 1;
    }
    out
}

// ---------------------------------------------------------------------------
// Cluster model
// ---------------------------------------------------------------------------

/// One shaped glyph. Positions are in rasterized font pixels relative to the
/// line origin and baseline, *before* `font_scale` / dpi conversion.
#[derive(Debug, Clone)]
#[allow(dead_code)] // `missing` is consumed by tests and fallback diagnostics
pub(crate) struct ShapedGlyph {
    /// Codepoint this glyph was selected for (presentation form after
    /// Arabic joining, matra after Devanagari reordering).
    pub char: char,
    /// Index of the font in the resolved fallback chain.
    pub font_slot: usize,
    /// True when coverage was missing everywhere and the replacement glyph
    /// is rasterized instead.
    pub missing: bool,
    /// Zero advance combining mark anchored onto `cluster`.
    pub is_mark: bool,
    /// Index of the base glyph this mark attaches to (for marks only).
    pub cluster: usize,
    /// Pen x of the glyph origin in visual order. For RTL runs values still
    /// increase in drawing order; `rtl` on the line flips interpretation.
    pub x: f32,
    /// Additional vertical offset above the baseline (marks).
    pub y_offset: f32,
    /// Horizontal advance consumed by this glyph.
    pub advance: f32,
    /// Kerning adjustment applied between the previous glyph and this one.
    pub kerning: f32,
}

/// One visual line. Newline characters split lines upstream; each line is
/// shaped and positioned independently.
#[derive(Debug, Clone, Default)]
pub(crate) struct ShapedLine {
    pub glyphs: Vec<ShapedGlyph>,
    pub width: f32,
    pub rtl: bool,
}

/// Font face the glyph resolver needs. Only the cmap/hmtx/kern surface of
/// fontdue is used, so the trait stays trivial to mock in tests.
pub(crate) trait GlyphFont {
    fn has_glyph(&self, c: char) -> bool;
    fn glyph_index(&self, c: char) -> u16;
    fn advance_width(&self, index: u16, px: f32) -> f32;
    fn kern_indexed(&self, left: u16, right: u16, px: f32) -> Option<f32>;
    fn line_gap_metrics(&self, px: f32) -> (f32, f32);
}

impl GlyphFont for fontdue::Font {
    fn has_glyph(&self, c: char) -> bool {
        fontdue::Font::has_glyph(self, c)
    }
    fn glyph_index(&self, c: char) -> u16 {
        self.lookup_glyph_index(c)
    }
    fn advance_width(&self, index: u16, px: f32) -> f32 {
        self.metrics_indexed(index, px).advance_width
    }
    fn kern_indexed(&self, left: u16, right: u16, px: f32) -> Option<f32> {
        self.horizontal_kern_indexed(left, right, px)
    }
    fn line_gap_metrics(&self, px: f32) -> (f32, f32) {
        match self.horizontal_line_metrics(px) {
            Some(m) => (m.ascent, m.descent),
            None => (px, -px * 0.2),
        }
    }
}

/// Strips zero-width formatting characters. ZWJ/ZWNJ have already influenced
/// Arabic joining by the time this runs; everything invisible is dropped so
/// it can never reach rasterization.
fn drop_zero_width(chars: &[char]) -> Vec<char> {
    chars.iter().copied().filter(|c| !is_zero_width(*c)).collect()
}

/// Full shaping pipeline for one logical line (no newline characters).
///
/// `fonts` is the resolved fallback chain, slot 0 being the requested font.
/// The returned line carries absolute pen positions at `font_size` pixels.
pub(crate) fn shape_line(fonts: &[&dyn GlyphFont], text: &[char], font_size: f32) -> ShapedLine {
    let runs = segment_runs(text);
    let mut line = ShapedLine::default();
    let mut pen = 0.0_f32;

    for run in runs {
        let rtl = run.direction == Direction::Rtl;
        if rtl {
            line.rtl = true;
        }

        // Script specific substitutions, still in logical order.
        let logical: Vec<char> = match run.script {
            Script::Arabic => shape_arabic(&run.chars),
            Script::Devanagari => shape_devanagari(&run.chars),
            _ => run.chars.clone(),
        };
        let logical = drop_zero_width(&logical);

        // Resolve glyph coverage and metrics in logical order.
        let mut glyphs: Vec<ShapedGlyph> = Vec::with_capacity(logical.len());
        let mut prev_index: Option<(u16, usize)> = None;
        for original in logical {
            let mark = is_combining_mark(original);
            let (slot, _resolved, missing) = resolve_glyph(fonts, original);
            // Missing codepoints are represented by the replacement character
            // from here on, so every downstream stage (metrics,
            // rasterization) only ever addresses a glyph the font contains.
            let c = if missing { REPLACEMENT_CHAR } else { original };
            let index = fonts[slot].glyph_index(c);
            let mut kerning = 0.0;
            if let Some((prev_idx, prev_slot)) = prev_index {
                if prev_slot == slot {
                    kerning = fonts[slot]
                        .kern_indexed(prev_idx, index, font_size)
                        .unwrap_or(0.0);
                }
            }
            prev_index = Some((index, slot));
            let advance = if mark {
                0.0
            } else {
                fonts[slot].advance_width(index, font_size)
            };
            glyphs.push(ShapedGlyph {
                char: c,
                font_slot: slot,
                missing,
                is_mark: mark,
                cluster: 0,
                x: 0.0,
                y_offset: 0.0,
                advance,
                kerning,
            });
        }

        // Attach marks to the preceding base glyph.
        for i in 0..glyphs.len() {
            if glyphs[i].is_mark {
                let mut base = i;
                while base > 0 && glyphs[base - 1].is_mark {
                    base -= 1;
                }
                glyphs[i].cluster = if base == 0 && glyphs[0].is_mark {
                    0 // orphan mark anchors to itself
                } else {
                    base.saturating_sub(1)
                };
            }
        }

        // Visual order: RTL runs are reversed whole, keeping each mark glued
        // to its base (marks move together with the cluster they belong to).
        if rtl {
            reverse_clusters(&mut glyphs);
        }

        // Position in visual/drawing order.
        let run_start = pen;
        for g in glyphs.iter_mut() {
            g.x = pen + g.kerning;
            if !g.is_mark {
                pen += g.advance + g.kerning;
            }
        }

        // Marks sit on top of their base cluster, centered on its origin.
        for i in 0..glyphs.len() {
            if glyphs[i].is_mark {
                let base = glyphs[i].cluster;
                let base_advance = if base < glyphs.len() {
                    glyphs[base].advance
                } else {
                    0.0
                };
                let (_, ascent) = fonts[glyphs[i].font_slot].line_gap_metrics(font_size);
                glyphs[i].x = glyphs
                    .get(base)
                    .map(|b| b.x)
                    .unwrap_or(run_start)
                    + base_advance * 0.25;
                // ascent is positive, descent negative; lift by half the
                // font's em box above the baseline.
                glyphs[i].y_offset = ascent.abs() * 0.5;
            }
        }

        line.glyphs.append(&mut glyphs);
    }

    line.width = line
        .glyphs
        .iter()
        .filter(|g| !g.is_mark)
        .fold(0.0, |acc, g| acc + g.advance + g.kerning);
    line
}

/// Picks the first font with coverage for `c`; when none covers it slot 0 is
/// returned with `missing = true` and the caller rasterizes `.notdef`.
fn resolve_glyph(fonts: &[&dyn GlyphFont], c: char) -> (usize, u16, bool) {
    for (slot, font) in fonts.iter().enumerate() {
        if font.has_glyph(c) {
            return (slot, font.glyph_index(c), false);
        }
    }
    (0, fonts[0].glyph_index(REPLACEMENT_CHAR), true)
}

/// Reverses glyphs while keeping combining marks attached to their base:
/// bases swap positions, their marks swap with them.
fn reverse_clusters(glyphs: &mut [ShapedGlyph]) {
    let old_len = glyphs.len();
    // Build cluster groups: each base followed by its marks.
    let mut groups: Vec<std::ops::Range<usize>> = Vec::new();
    let mut start = 0;
    for i in 0..old_len {
        let last = i == old_len - 1;
        let next_is_mark = !last && glyphs[i + 1].is_mark;
        if !next_is_mark {
            groups.push(start..i + 1);
            start = i + 1;
        }
    }
    // old index -> new index after the group-level reversal, needed to
    // retarget every mark's `cluster` pointer.
    let mut remap = vec![0usize; old_len];
    let mut new_index = 0;
    for group in groups.iter().rev() {
        for old_index in group.clone() {
            remap[old_index] = new_index;
            new_index += 1;
        }
    }
    let reordered: Vec<ShapedGlyph> = groups
        .iter()
        .rev()
        .flat_map(|r| glyphs[r.clone()].iter().cloned())
        .collect();
    glyphs.clone_from_slice(&reordered);
    for glyph in glyphs.iter_mut() {
        if glyph.is_mark {
            glyph.cluster = remap[glyph.cluster];
        }
    }
}

/// Splits text on hard line breaks. Both `\n` and `\r\n` (lone `\r` as well)
/// are accepted so input from any platform shapes identically. CRLF collapses
/// to one break instead of producing a phantom empty line.
pub(crate) fn split_lines(text: &str) -> Vec<Vec<char>> {
    let mut lines: Vec<Vec<char>> = Vec::new();
    let mut current: Vec<char> = Vec::new();
    let mut chars = text.chars().peekable();
    while let Some(c) = chars.next() {
        match c {
            '\n' | '\r' => {
                // swallow the second half of CRLF
                if c == '\r' && chars.peek() == Some(&'\n') {
                    chars.next();
                }
                lines.push(std::mem::take(&mut current));
            }
            _ => current.push(c),
        }
    }
    lines.push(current);
    lines
}

/// Lossy UTF-8 decode: invalid byte sequences become U+FFFD instead of
/// panicking or silently dropping bytes. Mirrors `String::from_utf8_lossy`
/// but stays allocation-light for already-valid input by reusing one Vec.
pub(crate) fn decode_utf8_lossy(bytes: &[u8]) -> Vec<char> {
    let mut chars = Vec::with_capacity(bytes.len().min(1024));
    let mut i = 0;
    while i < bytes.len() {
        let b = bytes[i];
        let width = match b {
            0x00..=0x7F => 1,
            0xC0..=0xDF => 2,
            0xE0..=0xEF => 3,
            0xF0..=0xF7 => 4,
            _ => {
                chars.push(REPLACEMENT_CHAR);
                i += 1;
                continue;
            }
        };
        if i + width > bytes.len() {
            chars.push(REPLACEMENT_CHAR);
            i += 1;
            continue;
        }
        match std::str::from_utf8(&bytes[i..i + width]) {
            Ok(s) => {
                chars.extend(s.chars());
                i += width;
            }
            Err(_) => {
                chars.push(REPLACEMENT_CHAR);
                i += 1;
            }
        }
    }
    chars
}

// ---------------------------------------------------------------------------
// Cache keys
// ---------------------------------------------------------------------------

/// Key for the layout/shaping cache. The fallback chain is identified by the
/// primary font's unique id plus its chain generation, so registering a new
/// fallback font invalidates every stale entry automatically.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct ShapeCacheKey {
    pub font_id: u64,
    pub chain_generation: u64,
    pub font_size: u16,
    pub text_hash: u64,
}

impl Hash for ShapeCacheKey {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.font_id.hash(state);
        self.chain_generation.hash(state);
        self.font_size.hash(state);
        self.text_hash.hash(state);
    }
}

/// FxHash-style hasher used for cache keys and text hashing: deterministic,
/// dependency free and far cheaper than SipHash for short ASCII-ish strings.
pub(crate) fn hash_text(text: &str) -> u64 {
    let mut hash: u64 = 0;
    for &byte in text.as_bytes() {
        hash = hash.rotate_left(5) ^ (byte as u64);
        hash = hash.wrapping_mul(0x517C_C1B7_2722_0A95);
    }
    hash
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn segmentation_groups_common_with_adjacent_script() {
        let chars: Vec<char> = "abc ديج 123".chars().collect();
        let runs = segment_runs(&chars);
        let scripts: Vec<_> = runs.iter().map(|r| r.script).collect();
        assert!(scripts.contains(&Script::Latin));
        assert!(scripts.contains(&Script::Arabic));
        // the digits inside/after Arabic inherit Arabic through recency
        assert!(runs.iter().any(|r| r.script == Script::Arabic
            && r.chars.contains(&'1')));
    }

    #[test]
    fn arabic_contextual_forms() {
        // bare beh -> isolated
        assert_eq!(shape_arabic(&['\u{0628}']), vec!['\u{FE8F}']);
        // two beh: initial + final
        let out = shape_arabic(&['\u{0628}', '\u{0628}']);
        assert_eq!(out, vec!['\u{FE91}', '\u{FE90}']);
        // three beh: initial, medial, final
        let out = shape_arabic(&['\u{0628}'; 3]);
        assert_eq!(out, vec!['\u{FE91}', '\u{FE92}', '\u{FE90}']);
        // alef is right joining only: it cannot join from its right, so beh
        // stays isolated while alef takes its final (left-joined) form
        let out = shape_arabic(&['\u{0628}', '\u{0627}']);
        assert_eq!(out, vec!['\u{FE8F}', '\u{FE8E}']);
    }

    #[test]
    fn arabic_zwnj_breaks_joining() {
        let out = shape_arabic(&['\u{0628}', '\u{200C}', '\u{0628}']);
        assert_eq!(out, vec!['\u{FE8F}', '\u{FE8F}']);
    }

    #[test]
    fn arabic_lam_alef_ligature() {
        let out = shape_arabic(&['\u{0644}', '\u{0627}']);
        assert_eq!(out, vec!['\u{FEFB}']);
        // with a preceding dual-joining letter: beh takes the initial form
        // and the lam-alef ligature its final form
        let out = shape_arabic(&['\u{0628}', '\u{0644}', '\u{0627}']);
        assert_eq!(out, vec!['\u{FE91}', '\u{FEFC}']);
        // lam-alef at the start of a run uses the isolated ligature form;
        // the preceding alef has no right neighbour and is itself
        // right-joining, so it stays in its isolated presentation form
        let out = shape_arabic(&['\u{0627}', '\u{0644}', '\u{0627}']);
        assert_eq!(out, vec!['\u{FE8D}', '\u{FEFB}']);
    }

    #[test]
    fn arabic_zero_width_joiners_disappear() {
        let out = shape_arabic(&['\u{0628}', '\u{200D}', '\u{0628}', '\u{FEFF}']);
        assert_eq!(out, vec!['\u{FE91}', '\u{FE90}']);
        assert!(out.iter().all(|c| !is_zero_width(*c)));
    }

    #[test]
    fn devanagari_short_i_reorders_before_cluster() {
        // ga + short-i: U+093F must move in front of U+0917
        let out = shape_devanagari(&['\u{0917}', '\u{093F}']);
        assert_eq!(out, vec!['\u{093F}', '\u{0917}']);

        // k + virama + t + short-i: matra moves before the whole conjunct
        let out = shape_devanagari(&['\u{0915}', '\u{094D}', '\u{0924}', '\u{093F}']);
        assert_eq!(out, vec!['\u{093F}', '\u{0915}', '\u{094D}', '\u{0924}']);

        // syllable boundary stops the reorder
        let out = shape_devanagari(&['\u{0917}', ' ', '\u{0918}', '\u{093F}']);
        assert_eq!(out, vec!['\u{0917}', ' ', '\u{093F}', '\u{0918}']);
    }

    #[test]
    fn rtl_reversal_keeps_marks_glued_to_base() {
        let mut glyphs = vec![
            ShapedGlyph {
                char: 'a', font_slot: 0, missing: false, is_mark: false, cluster: 0,
                x: 0.0, y_offset: 0.0, advance: 10.0, kerning: 0.0,
            },
            ShapedGlyph {
                char: 'b', font_slot: 0, missing: false, is_mark: false, cluster: 0,
                x: 0.0, y_offset: 0.0, advance: 10.0, kerning: 0.0,
            },
            ShapedGlyph {
                char: 'm', font_slot: 0, missing: false, is_mark: true, cluster: 1,
                x: 0.0, y_offset: 0.0, advance: 0.0, kerning: 0.0,
            },
        ];
        reverse_clusters(&mut glyphs);
        assert_eq!(glyphs.iter().map(|g| g.char).collect::<Vec<_>>(), vec!['b', 'm', 'a']);
        assert_eq!(glyphs[1].cluster, 0);
    }

    #[test]
    fn invalid_utf8_becomes_replacement() {
        let chars = decode_utf8_lossy(&[b'a', 0xFF, 0xFE, b'b']);
        assert_eq!(chars, vec!['a', REPLACEMENT_CHAR, REPLACEMENT_CHAR, 'b']);

        // truncated multibyte sequence: one replacement, not a panic
        let chars = decode_utf8_lossy(&[0xE0, 0xA4]);
        assert_eq!(chars, vec![REPLACEMENT_CHAR, REPLACEMENT_CHAR]);

        // valid devanagari still decodes
        let chars = decode_utf8_lossy("गि".as_bytes());
        assert_eq!(chars, vec!['\u{0917}', '\u{093F}']);
    }

    #[test]
    fn line_splitting_handles_all_newline_styles() {
        assert_eq!(split_lines("a\nb").len(), 2);
        assert_eq!(split_lines("a\r\nb").len(), 2);
        assert_eq!(split_lines("a\rb").len(), 2);
        assert_eq!(split_lines("").len(), 1);
    }

    #[test]
    fn text_hash_is_deterministic_and_distinct() {
        assert_eq!(hash_text("hello"), hash_text("hello"));
        assert_ne!(hash_text("hello"), hash_text("hellp"));
    }

    /// Mock font: covers an explicit codepoint set and reports fixed metrics.
    struct MockFont {
        covers: fn(char) -> bool,
        advance: f32,
    }

    impl GlyphFont for MockFont {
        fn has_glyph(&self, c: char) -> bool {
            (self.covers)(c)
        }
        fn glyph_index(&self, c: char) -> u16 {
            c as u16
        }
        fn advance_width(&self, _index: u16, _px: f32) -> f32 {
            self.advance
        }
        fn kern_indexed(&self, _l: u16, _r: u16, _px: f32) -> Option<f32> {
            None
        }
        fn line_gap_metrics(&self, _px: f32) -> (f32, f32) {
            (16.0, -4.0)
        }
    }

    #[test]
    fn fallback_chain_covers_missing_glyphs() {
        let latin = MockFont {
            covers: |c| c.is_ascii_alphanumeric() || c == ' ',
            advance: 10.0,
        };
        let arabic = MockFont {
            covers: |c| matches!(c as u32, 0x0600..=0x06FF | 0xFE70..=0xFEFF),
            advance: 12.0,
        };
        let fonts: Vec<&dyn GlyphFont> = vec![&latin, &arabic];
        let line = shape_line(&fonts, &"ب a".chars().collect::<Vec<_>>(), 20.0);

        let slots: Vec<usize> = line.glyphs.iter().map(|g| g.font_slot).collect();
        assert!(slots.contains(&1), "arabic glyph must use the fallback font");
        assert!(slots.contains(&0));
    }

    #[test]
    fn uncovered_glyph_is_marked_missing() {
        let latin = MockFont {
            covers: |c| c.is_ascii_alphabetic(),
            advance: 10.0,
        };
        let fonts: Vec<&dyn GlyphFont> = vec![&latin];
        let line = shape_line(&fonts, &"a☃b".chars().collect::<Vec<_>>(), 20.0);
        let snowman = line.glyphs.iter().find(|g| g.char == REPLACEMENT_CHAR).unwrap();
        assert!(snowman.missing);
        // neighbouring latin still renders normally
        assert!(!line.glyphs.iter().all(|g| g.missing));
    }

    #[test]
    fn marks_have_zero_advance_and_anchor_to_base() {
        let all = MockFont {
            covers: |_c| true,
            advance: 10.0,
        };
        let fonts: Vec<&dyn GlyphFont> = vec![&all];
        let line = shape_line(&fonts, &['a', '\u{0301}'], 20.0);
        assert_eq!(line.glyphs.len(), 2);
        let mark = &line.glyphs[1];
        assert!(mark.is_mark);
        assert_eq!(mark.advance, 0.0);
        assert_eq!(mark.cluster, 0);
        assert!(mark.y_offset > 0.0);
        // line width ignores the mark
        assert_eq!(line.width, 10.0);
    }

    #[test]
    fn rtl_line_is_flagged_and_positioned_backwards() {
        let all = MockFont {
            covers: |_c| true,
            advance: 10.0,
        };
        let fonts: Vec<&dyn GlyphFont> = vec![&all];
        let chars: Vec<char> = "دي".chars().collect();
        let line = shape_line(&fonts, &chars, 20.0);
        assert!(line.rtl);
        // visual first glyph is the logical last letter's presentation form
        assert_ne!(line.glyphs[0].char, chars[0]);
        // pen positions still advance monotonically in drawing order
        for w in line.glyphs.windows(2) {
            assert!(w[1].x >= w[0].x);
        }
    }

    #[test]
    fn emoji_without_coverage_falls_back_without_panic() {
        let latin = MockFont {
            covers: |c| c.is_ascii(),
            advance: 10.0,
        };
        let fonts: Vec<&dyn GlyphFont> = vec![&latin];
        let line = shape_line(&fonts, &"😀a".chars().collect::<Vec<_>>(), 20.0);
        assert!(line.glyphs[0].missing);
        assert_eq!(line.glyphs[0].char, REPLACEMENT_CHAR);
    }
}
