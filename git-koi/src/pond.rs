//! A wide pond around the koi, built from the rest of the account's activity.
//!
//! The pool floor is the last year's contribution graph as a mosaic. A fry swims for
//! each language, in GitHub's color for it. Pellets are the last seven days'
//! contributions, and the koi eats them as it passes. A lily pad floats for each year
//! on GitHub, and they bloom while a weekly streak runs. The koi still moves as the
//! day's [`Behavior`] says. Identical inputs produce identical pixels without I/O or
//! clock reads, and every frame count closes the loop seamlessly.

use std::f64::consts::{PI, TAU};

use chrono::{Datelike, Days};

use crate::behavior::{Behavior, select_behavior};
use crate::coat::KoiBody;
use crate::history::{AccountSnapshot, summarize_contributions};
use crate::koi::{KoiDevelopment, KoiDevelopmentError, RecentStreak, derive_koi_development};
use crate::motion::{Course, Stadium, SwimPose, calculate_swim_pose};
use crate::render::{FRAME_HEIGHT_PIXELS, FRAME_WIDTH_PIXELS, render_placed_koi};

/// Width of the native scene; exports scale it up exactly.
pub const POND_WIDTH_PIXELS: u32 = 520;
/// Height of the native scene.
pub const POND_HEIGHT_PIXELS: u32 = 208;

const WIDTH: i32 = POND_WIDTH_PIXELS as i32;
const HEIGHT: i32 = POND_HEIGHT_PIXELS as i32;
const TRACK: Stadium = Stadium {
    center: [260.0, 104.0],
    half_straight_pixels: 140.0,
    radius_pixels: 70.0,
};
/// Frames per pose loop of a lap style; the pond holds two, for one slow lap.
const POSE_LOOP_FRAMES: u32 = 120;
const LAP_FRAMES: u32 = 2 * POSE_LOOP_FRAMES;
const REST_FRAMES: u32 = 180;

const MOSAIC_WEEKS: i32 = 53;
const MOSAIC_DAYS: i32 = 7;
const TILE_PIXELS: i32 = 8;
const TILE_STEP_PIXELS: i32 = 9;
const MOSAIC_LEFT: i32 = (WIDTH - MOSAIC_WEEKS * TILE_STEP_PIXELS + 1) / 2;
const MOSAIC_TOP: i32 = (HEIGHT - MOSAIC_DAYS * TILE_STEP_PIXELS + 1) / 2;
const TILE_SHADES: [[u8; 3]; 5] = [
    [20, 72, 90],
    [24, 92, 94],
    [32, 114, 96],
    [50, 140, 98],
    [88, 170, 104],
];

const WATER_SHADES: [[u8; 3]; 3] = [[15, 58, 76], [17, 66, 84], [20, 74, 92]];
const STONE_SHADES: [[u8; 3]; 4] = [[50, 62, 68], [58, 70, 76], [66, 78, 82], [48, 74, 62]];
const CORNER_RADIUS_PIXELS: f64 = 14.0;
const SHORE_WIDTH_PIXELS: f64 = 5.0;
const SHADOW_OFFSET_PIXELS: i32 = 4;
const SHADOW_STRENGTH: f64 = 0.68;

const MAX_FRY: usize = 12;
const MAX_PELLETS: u64 = 24;
/// Frames a pellet floats before the koi arrives, and how long its ripples last.
const PELLET_FLOAT_FRAMES: u32 = 80;
const RIPPLE_FRAMES: u32 = 16;
const MOUTH_OPEN_FRAMES: u32 = 10;
const SPARKLE_COUNT: u32 = 26;
const MAX_LILY_PADS: usize = 10;
/// Pads sit in the corners and along the edges, where the koi's track leaves water.
const LILY_PAD_SPOTS: [[i32; 2]; MAX_LILY_PADS] = [
    [30, 28],
    [490, 180],
    [492, 30],
    [28, 180],
    [62, 22],
    [458, 186],
    [458, 24],
    [62, 186],
    [18, 104],
    [502, 104],
];

/// One account's pond, ready to draw any frame.
pub struct Pond {
    koi_development: KoiDevelopment,
    koi_body: KoiBody,
    behavior: Behavior,
    /// The still water, shore, and mosaic.
    background: Vec<[u8; 4]>,
    /// The still lily pads, which float above everything that swims.
    canopy: Vec<[u8; 4]>,
    fry: Vec<Fry>,
    pellets: Vec<Pellet>,
    sparkles: Vec<Sparkle>,
}

struct Fry {
    colors: [[u8; 4]; 3],
    length_pixels: f64,
    start_share: f64,
    lap_count: f64,
    side_offset_pixels: f64,
    wobble_cycles: f64,
}

struct Pellet {
    position: [f64; 2],
    /// The frame the koi reaches it; `None` when the koi is resting and leaves it afloat.
    eaten_at: Option<u32>,
    phase_frames: u32,
}

struct Sparkle {
    position: [i32; 2],
    phase_frames: u32,
}

impl Pond {
    /// Derive everything the pond shows. A snapshot the koi cannot be drawn from fails
    /// the same way here.
    pub fn new(
        account_snapshot: &AccountSnapshot,
        koi_body: KoiBody,
    ) -> Result<Self, KoiDevelopmentError> {
        let koi_development = derive_koi_development(account_snapshot)?;
        let contribution_summary = summarize_contributions(account_snapshot)
            .map_err(KoiDevelopmentError::ContributionHistory)?;
        let behavior = select_behavior(&koi_development);
        let observed_on = contribution_summary.observed_on;
        let seed = koi_development.identity_seed;

        let mut background = draw_water(seed);
        let last_column_day = observed_on.weekday().num_days_from_sunday() as i32;
        let year_counts: Vec<u32> = (0..u64::from((MOSAIC_WEEKS * MOSAIC_DAYS) as u32))
            .filter_map(|days_back| {
                observed_on
                    .checked_sub_days(Days::new(days_back))
                    .and_then(|date| contribution_summary.daily_counts.get(&date).copied())
            })
            .collect();
        let thresholds = calculate_tile_thresholds(&year_counts);
        for column in 0..MOSAIC_WEEKS {
            for row in 0..MOSAIC_DAYS {
                let days_back = (MOSAIC_WEEKS - 1 - column) * MOSAIC_DAYS + last_column_day - row;
                if days_back < 0 {
                    continue;
                }
                let count = observed_on
                    .checked_sub_days(Days::new(days_back as u64))
                    .and_then(|date| contribution_summary.daily_counts.get(&date).copied())
                    .unwrap_or(0);
                let level = thresholds
                    .iter()
                    .filter(|threshold| count >= **threshold)
                    .count();
                draw_tile(
                    &mut background,
                    MOSAIC_LEFT + column * TILE_STEP_PIXELS,
                    MOSAIC_TOP + row * TILE_STEP_PIXELS,
                    TILE_SHADES[level],
                );
            }
        }

        let first_day = contribution_summary.daily_counts.keys().next().copied();
        let year_count = first_day.map_or(0, |first_day| {
            (observed_on - first_day).num_days().max(0) as usize / 365 + 1
        });
        let streak_weeks = match &koi_development.streaks.recent {
            Some(RecentStreak::Current(weeks)) => weeks.len(),
            _ => 0,
        };
        let mut canopy = vec![[0; 4]; (WIDTH * HEIGHT) as usize];
        for (pad_index, spot) in LILY_PAD_SPOTS.iter().take(year_count).enumerate() {
            // One bloom per four weeks of the running streak, starting with the first.
            let is_blooming = pad_index < streak_weeks.div_ceil(4);
            draw_lily_pad(&mut canopy, *spot, pad_index, is_blooming);
        }

        let mut language_shares: Vec<(&String, f64)> = koi_development
            .language_shares
            .iter()
            .map(|(name, share)| (name, *share))
            .collect();
        language_shares.sort_by(|left, right| right.1.total_cmp(&left.1).then(left.0.cmp(right.0)));
        language_shares.truncate(MAX_FRY);
        let fry_count = language_shares.len().max(1) as f64;
        let fry = language_shares
            .iter()
            .enumerate()
            .map(|(fry_index, (name, share))| {
                let color = koi_development
                    .language_colors
                    .get(*name)
                    .copied()
                    .flatten()
                    .map_or([170, 180, 190], |language_color| language_color.0);
                Fry {
                    colors: shade_fry(color),
                    length_pixels: 15.0 + 13.0 * share.sqrt(),
                    start_share: 0.16 + 0.8 * fry_index as f64 / fry_count,
                    lap_count: if fry_index % 3 == 0 { 2.0 } else { 1.0 },
                    side_offset_pixels: [-12.0, 26.0, -6.0, 34.0, 10.0, -16.0][fry_index % 6],
                    wobble_cycles: 3.0 + (fry_index % 3) as f64,
                }
            })
            .collect();

        let frame_count = frames_for(behavior);
        let pellet_count = koi_development
            .recent_activity
            .recent_contributions
            .min(MAX_PELLETS) as u32;
        let pellets = (0..pellet_count)
            .map(|pellet_index| {
                let eat_frame = (pellet_index * frame_count + frame_count / 2) / pellet_count;
                let jitter = hash(seed, u64::from(pellet_index));
                if is_resting(behavior) {
                    Pellet {
                        position: [
                            40.0 + (jitter % 440) as f64,
                            22.0 + ((jitter >> 16) % 164) as f64,
                        ],
                        eaten_at: None,
                        phase_frames: eat_frame,
                    }
                } else {
                    let head = calculate_lap_pose(behavior, eat_frame);
                    let distance = head_distance(&head);
                    let [x, y] = TRACK.calculate_point(distance + 4.0);
                    Pellet {
                        position: [
                            x + (jitter % 5) as f64 - 2.0,
                            y + ((jitter >> 8) % 5) as f64 - 2.0,
                        ],
                        eaten_at: Some(eat_frame),
                        phase_frames: eat_frame,
                    }
                }
            })
            .collect();

        let sparkles = (0..SPARKLE_COUNT)
            .map(|sparkle_index| {
                let mixed = hash(seed ^ 0x51ab, u64::from(sparkle_index));
                Sparkle {
                    position: [
                        12 + (mixed % (WIDTH as u64 - 24)) as i32,
                        12 + ((mixed >> 20) % (HEIGHT as u64 - 24)) as i32,
                    ],
                    phase_frames: (mixed >> 40) as u32 % frame_count,
                }
            })
            .collect();

        Ok(Self {
            koi_development,
            koi_body,
            behavior,
            background,
            canopy,
            fry,
            pellets,
            sparkles,
        })
    }

    /// Frames in the pond's loop.
    pub fn frame_count(&self) -> u32 {
        frames_for(self.behavior)
    }

    /// The fry swimming today, one per language up to the limit.
    pub fn fry_count(&self) -> usize {
        self.fry.len()
    }

    /// The pellets in the pond today.
    pub fn pellet_count(&self) -> usize {
        self.pellets.len()
    }

    /// One frame's pixels, row by row from the top left. Frame numbers wrap around the
    /// loop. Every pixel is opaque except the rounded corners outside the shore.
    pub fn render_frame(&self, frame_index: u32) -> Vec<[u8; 4]> {
        let frame_count = self.frame_count();
        let frame = frame_index % frame_count;
        let mut pixels = self.background.clone();
        for sparkle in &self.sparkles {
            draw_sparkle(&mut pixels, sparkle, frame, frame_count);
        }
        let mut swim_mask = vec![false; pixels.len()];
        for fry in &self.fry {
            draw_fry(&mut pixels, &mut swim_mask, fry, frame, frame_count);
        }
        for pellet in &self.pellets {
            draw_pellet(&mut pixels, pellet, frame, frame_count);
        }
        self.draw_koi(&mut pixels, &mut swim_mask, frame);
        for (pixel, canopy_pixel) in pixels.iter_mut().zip(&self.canopy) {
            if canopy_pixel[3] != 0 {
                *pixel = *canopy_pixel;
            }
        }
        pixels
    }

    fn draw_koi(&self, pixels: &mut [[u8; 4]], swim_mask: &mut [bool], frame: u32) {
        let (mut swim_pose, world_offset) = if is_resting(self.behavior) {
            // The body trails left of the head, so the head rests right of center.
            let pose = calculate_swim_pose(self.behavior, frame);
            (pose, [320, 104])
        } else {
            (calculate_lap_pose(self.behavior, frame), [0, 0])
        };
        if let Some(gap) = self
            .pellets
            .iter()
            .filter_map(|pellet| pellet.eaten_at)
            .map(|eaten_at| (eaten_at + self.frame_count() - frame) % self.frame_count())
            .min()
            && gap < MOUTH_OPEN_FRAMES
        {
            let opening = 1.0 - f64::from(gap) / f64::from(MOUTH_OPEN_FRAMES);
            swim_pose.mouth_opening = swim_pose.mouth_opening.max(opening);
        }
        let (koi_frame, scene_offset) = render_placed_koi(
            &self.koi_development,
            self.koi_body,
            &swim_pose,
            world_offset,
        );
        let koi_pixels = koi_frame.as_rgba_bytes().as_chunks::<4>().0;
        let visit = |visitor: &mut dyn FnMut(i32, i32, [u8; 4])| {
            for canvas_y in 0..FRAME_HEIGHT_PIXELS as i32 {
                for canvas_x in 0..FRAME_WIDTH_PIXELS as i32 {
                    let color =
                        koi_pixels[(canvas_y * FRAME_WIDTH_PIXELS as i32 + canvas_x) as usize];
                    if color[3] != 0 {
                        visitor(
                            canvas_x + scene_offset[0],
                            canvas_y + scene_offset[1],
                            color,
                        );
                    }
                }
            }
        };
        visit(&mut |x, y, _| {
            darken(
                pixels,
                swim_mask,
                x + SHADOW_OFFSET_PIXELS,
                y + SHADOW_OFFSET_PIXELS,
            )
        });
        visit(&mut |x, y, color| put(pixels, x, y, color));
    }
}

fn frames_for(behavior: Behavior) -> u32 {
    if behavior == Behavior::Rest {
        REST_FRAMES
    } else {
        LAP_FRAMES
    }
}

fn is_resting(behavior: Behavior) -> bool {
    matches!(behavior, Behavior::Rest | Behavior::DeepRest)
}

/// A lap pose on the pond's track. Surface feeding swims too: the pond's food comes to
/// it. A cruise takes one lap over the loop, a dart two.
fn calculate_lap_pose(behavior: Behavior, frame: u32) -> SwimPose {
    let lap_behavior = if behavior == Behavior::Dart {
        Behavior::Dart
    } else {
        Behavior::Cruise
    };
    let laps_per_pose_loop = if behavior == Behavior::Dart { 2.0 } else { 1.0 };
    let pose = calculate_swim_pose(lap_behavior, frame);
    let Course::Lap { angle_radians } = pose.course else {
        unreachable!("lap styles swim laps");
    };
    let pose_loop_index = f64::from(frame / POSE_LOOP_FRAMES);
    let turned_laps = (angle_radians + PI / 2.0) / TAU;
    let share = (pose_loop_index * laps_per_pose_loop + turned_laps) / 2.0;
    SwimPose {
        course: Course::Stadium {
            stadium: TRACK,
            distance_pixels: share * TRACK.perimeter_pixels(),
        },
        ..pose
    }
}

fn head_distance(pose: &SwimPose) -> f64 {
    match pose.course {
        Course::Stadium {
            distance_pixels, ..
        } => distance_pixels,
        _ => 0.0,
    }
}

/// Counts at which a tile reaches each shade above empty: any activity, then the
/// quartiles of active days, as GitHub's own graph grades them.
fn calculate_tile_thresholds(year_counts: &[u32]) -> [u32; 4] {
    let mut active: Vec<u32> = year_counts.iter().copied().filter(|c| *c > 0).collect();
    active.sort_unstable();
    let quartile = |share: usize| {
        active
            .get(active.len() * share / 4)
            .copied()
            .unwrap_or(u32::MAX)
    };
    [
        1,
        quartile(1).max(2),
        quartile(2).max(3),
        quartile(3).max(4),
    ]
}

fn hash(seed: u64, value: u64) -> u64 {
    let mut mixed = seed
        .wrapping_mul(0x9e37_79b9_7f4a_7c15)
        .wrapping_add(value.wrapping_mul(0xbf58_476d_1ce4_e5b9));
    mixed ^= mixed >> 29;
    mixed = mixed.wrapping_mul(0x94d0_49bb_1331_11eb);
    mixed ^ (mixed >> 32)
}

fn opaque([red, green, blue]: [u8; 3]) -> [u8; 4] {
    [red, green, blue, 255]
}

fn put(pixels: &mut [[u8; 4]], x: i32, y: i32, color: [u8; 4]) {
    if (0..WIDTH).contains(&x) && (0..HEIGHT).contains(&y) {
        let pixel = &mut pixels[(y * WIDTH + x) as usize];
        if pixel[3] != 0 {
            *pixel = color;
        }
    }
}

/// Darken a pixel once, however many shapes cast a shadow on it.
fn darken(pixels: &mut [[u8; 4]], swim_mask: &mut [bool], x: i32, y: i32) {
    if (0..WIDTH).contains(&x) && (0..HEIGHT).contains(&y) {
        let pixel_index = (y * WIDTH + x) as usize;
        if pixels[pixel_index][3] != 0 && !swim_mask[pixel_index] {
            swim_mask[pixel_index] = true;
            for channel in &mut pixels[pixel_index][..3] {
                *channel = (f64::from(*channel) * SHADOW_STRENGTH) as u8;
            }
        }
    }
}

/// How far inside the rounded pond a pixel is; negative outside it.
fn distance_inside(x: i32, y: i32) -> f64 {
    let half = [f64::from(WIDTH) / 2.0, f64::from(HEIGHT) / 2.0];
    let outside = [
        (f64::from(x) + 0.5 - half[0]).abs() - half[0] + CORNER_RADIUS_PIXELS,
        (f64::from(y) + 0.5 - half[1]).abs() - half[1] + CORNER_RADIUS_PIXELS,
    ];
    CORNER_RADIUS_PIXELS
        - outside[0].max(0.0).hypot(outside[1].max(0.0))
        - outside[0].max(outside[1]).min(0.0)
}

fn draw_water(seed: u64) -> Vec<[u8; 4]> {
    let mut pixels = Vec::with_capacity((WIDTH * HEIGHT) as usize);
    for y in 0..HEIGHT {
        for x in 0..WIDTH {
            let inside = distance_inside(x, y);
            if inside < 0.0 {
                pixels.push([0; 4]);
            } else if inside < SHORE_WIDTH_PIXELS {
                // Stones along the shore, each one a few pixels of one shade.
                let stone = hash(
                    seed,
                    u64::from((x / 5) as u32) << 16 | u64::from((y / 4) as u32),
                );
                pixels.push(opaque(STONE_SHADES[(stone % 4) as usize]));
            } else {
                // Soft light and dark bands, dithered between three shades.
                let (fx, fy) = (f64::from(x), f64::from(y));
                let band = 0.5
                    + 0.25 * (fx * 0.021 + fy * 0.05 + (fy * 0.03).sin() * 2.0).sin()
                    + 0.25 * (fx * 0.047 - fy * 0.031).sin();
                let dither = f64::from(BAYER[(y & 3) as usize][(x & 3) as usize]) / 16.0;
                let shade = ((band * 2.0 + dither - 0.25).floor().max(0.0) as usize).min(2);
                let edge_shadow = inside < SHORE_WIDTH_PIXELS + 3.0;
                let color = WATER_SHADES[shade];
                pixels.push(opaque(if edge_shadow {
                    color.map(|channel| (f64::from(channel) * 0.7) as u8)
                } else {
                    color
                }));
            }
        }
    }
    pixels
}

const BAYER: [[u8; 4]; 4] = [[0, 8, 2, 10], [12, 4, 14, 6], [3, 11, 1, 9], [15, 7, 13, 5]];

fn draw_tile(pixels: &mut [[u8; 4]], left: i32, top: i32, shade: [u8; 3]) {
    let highlight = shade.map(|channel| channel.saturating_add(14));
    for y in top..top + TILE_PIXELS {
        for x in left..left + TILE_PIXELS {
            let is_corner = (x == left || x == left + TILE_PIXELS - 1)
                && (y == top || y == top + TILE_PIXELS - 1);
            if !is_corner {
                put(
                    pixels,
                    x,
                    y,
                    opaque(if y == top { highlight } else { shade }),
                );
            }
        }
    }
}

fn draw_lily_pad(
    canopy: &mut [[u8; 4]],
    [center_x, center_y]: [i32; 2],
    index: usize,
    bloom: bool,
) {
    const PAD_DARK: [u8; 3] = [34, 104, 68];
    const PAD_MID: [u8; 3] = [52, 136, 80];
    const PAD_LIGHT: [u8; 3] = [86, 170, 94];
    const PETAL: [u8; 3] = [244, 150, 186];
    const PETAL_LIGHT: [u8; 3] = [252, 206, 222];
    const HEART: [u8; 3] = [250, 214, 96];
    let radius = 11.0 + (index % 3) as f64 * 1.5;
    let notch_angle = index as f64 * 2.4 + 0.6;
    let reach = radius.ceil() as i32;
    for y in -reach..=reach {
        for x in -reach..=reach {
            let (fx, fy) = (f64::from(x), f64::from(y));
            let distance = fx.hypot(fy);
            let angle_gap = (fy.atan2(fx) - notch_angle + PI).rem_euclid(TAU) - PI;
            if distance > radius || angle_gap.abs() < 0.28 {
                continue;
            }
            let color = if distance > radius - 1.5 {
                PAD_DARK
            } else if (angle_gap * 5.0).sin().abs() < 0.14 {
                PAD_LIGHT
            } else {
                PAD_MID
            };
            let pixel_index = ((center_y + y) * WIDTH + center_x + x) as usize;
            if (0..WIDTH).contains(&(center_x + x))
                && (0..HEIGHT).contains(&(center_y + y))
                && distance_inside(center_x + x, center_y + y) >= SHORE_WIDTH_PIXELS
            {
                canopy[pixel_index] = opaque(color);
            }
        }
    }
    if bloom {
        for y in -6_i32..=6 {
            for x in -6_i32..=6 {
                let (fx, fy) = (f64::from(x), f64::from(y));
                let distance = fx.hypot(fy);
                let petal_reach = 4.0 + 2.2 * (fy.atan2(fx) * 4.0).cos().abs();
                let color = if distance < 1.8 {
                    HEART
                } else if distance < 3.4 {
                    PETAL_LIGHT
                } else if distance < petal_reach {
                    PETAL
                } else {
                    continue;
                };
                let (px, py) = (center_x + x, center_y + y);
                if (0..WIDTH).contains(&px) && (0..HEIGHT).contains(&py) {
                    canopy[(py * WIDTH + px) as usize] = opaque(color);
                }
            }
        }
    }
}

/// A fry's light, main, and dark shades, lifted until a dark language color still shows
/// against the water.
fn shade_fry(color: [u8; 3]) -> [[u8; 4]; 3] {
    const MIN_LUMINANCE: f64 = 135.0;
    let channels = color.map(f64::from);
    let luminance = 0.3 * channels[0] + 0.59 * channels[1] + 0.11 * channels[2];
    let lift = ((MIN_LUMINANCE - luminance) / (255.0 - luminance)).clamp(0.0, 1.0);
    let main = channels.map(|channel| channel + (255.0 - channel) * lift);
    let mix = |target: f64, share: f64| {
        let [red, green, blue] = main.map(|channel| (channel + (target - channel) * share) as u8);
        [red, green, blue, 255]
    };
    [mix(255.0, 0.45), mix(0.0, 0.0), mix(0.0, 0.4)]
}

fn draw_fry(
    pixels: &mut [[u8; 4]],
    swim_mask: &mut [bool],
    fry: &Fry,
    frame: u32,
    frame_count: u32,
) {
    const TAIL_WAG_CYCLES: f64 = 32.0;
    const WOBBLE_PIXELS: f64 = 5.0;
    let progress = f64::from(frame) / f64::from(frame_count);
    let perimeter = TRACK.perimeter_pixels();
    let distance = (fry.start_share + fry.lap_count * progress) * perimeter;
    let [track_x, track_y] = TRACK.calculate_point(distance);
    let [ahead_x, ahead_y] = TRACK.calculate_point(distance + 1.0);
    let forward = [ahead_x - track_x, ahead_y - track_y];
    let side = fry.side_offset_pixels
        + WOBBLE_PIXELS * (TAU * fry.wobble_cycles * progress + fry.start_share * TAU).sin();
    let center = [track_x - forward[1] * side, track_y + forward[0] * side];
    let wag = (TAU * TAIL_WAG_CYCLES * progress + fry.start_share * TAU).sin();
    let length = fry.length_pixels;
    let half_width = (length * 0.24).max(3.0);
    let tail_length = length * 0.4;
    let reach = (length + tail_length).ceil() as i32 + 2;
    let mut shape = Vec::new();
    for dy in -reach..=reach {
        for dx in -reach..=reach {
            let offset = [f64::from(dx) + 0.5, f64::from(dy) + 0.5];
            // Along runs back from the snout; across runs to the fish's right.
            let along = -(offset[0] * forward[0] + offset[1] * forward[1]);
            let across = -offset[0] * forward[1] + offset[1] * forward[0];
            let bend = wag * 1.8 * (along / (length + tail_length)).clamp(0.0, 1.0).powi(2);
            let bent_across = across - bend;
            let body_share = along / length;
            let color = if (0.0..1.0).contains(&body_share) {
                let width = half_width * (PI * body_share.powf(0.7)).sin().max(0.18);
                if bent_across.abs() > width {
                    continue;
                }
                if (along - 2.6).abs() < 1.0 && (bent_across.abs() - half_width * 0.62).abs() < 0.8
                {
                    [14, 24, 30, 255]
                } else if bent_across.abs() > width - 0.9 {
                    fry.colors[2]
                } else if bent_across < -0.2 {
                    fry.colors[0]
                } else {
                    fry.colors[1]
                }
            } else if (0.0..1.0).contains(&((along - length) / tail_length)) {
                let tail_share = (along - length) / tail_length;
                if bent_across.abs() > 0.6 + tail_share * half_width * 1.1 {
                    continue;
                }
                fry.colors[2]
            } else {
                continue;
            };
            shape.push((
                center[0].floor() as i32 + dx,
                center[1].floor() as i32 + dy,
                color,
            ));
        }
    }
    for (x, y, _) in &shape {
        darken(pixels, swim_mask, x + 2, y + 2);
    }
    for (x, y, color) in shape {
        put(pixels, x, y, color);
    }
}

fn draw_pellet(pixels: &mut [[u8; 4]], pellet: &Pellet, frame: u32, frame_count: u32) {
    const PELLET_LIGHT: [u8; 4] = [255, 214, 120, 255];
    const PELLET_DARK: [u8; 4] = [214, 136, 54, 255];
    const RIPPLE: [u8; 4] = [120, 190, 200, 255];
    let [x, y] = pellet.position.map(|axis| axis.floor() as i32);
    let draw_ripple = |pixels: &mut [[u8; 4]], age: u32| {
        let radius = 4.0 + 10.0 * f64::from(age) / f64::from(RIPPLE_FRAMES);
        let dots = 16;
        for dot in 0..dots {
            // The ring thins out as it spreads.
            if age > RIPPLE_FRAMES / 2 && dot % 2 == 1 {
                continue;
            }
            let angle = TAU * f64::from(dot) / f64::from(dots);
            put(
                pixels,
                x + (radius * angle.cos()).round() as i32,
                y + (radius * angle.sin()).round() as i32,
                RIPPLE,
            );
        }
    };
    let draw_body = |pixels: &mut [[u8; 4]]| {
        for dy in -2_i32..=2 {
            for dx in -2_i32..=2 {
                if dx.abs() + dy.abs() < 4 {
                    put(pixels, x + dx, y + dy, PELLET_DARK);
                }
            }
        }
        for (dx, dy) in [(-1, -1), (0, -1), (-1, 0), (0, 0)] {
            put(pixels, x + dx, y + dy, PELLET_LIGHT);
        }
    };
    match pellet.eaten_at {
        Some(eaten_at) => {
            let until_eaten = (eaten_at + frame_count - frame) % frame_count;
            let since_eaten = (frame + frame_count - eaten_at) % frame_count;
            if until_eaten <= PELLET_FLOAT_FRAMES {
                let age = PELLET_FLOAT_FRAMES - until_eaten;
                if age < RIPPLE_FRAMES {
                    draw_ripple(pixels, age);
                }
                if until_eaten > 0 {
                    draw_body(pixels);
                }
            } else if since_eaten < RIPPLE_FRAMES {
                draw_ripple(pixels, since_eaten);
            }
        }
        None => {
            draw_body(pixels);
            let age = (frame + pellet.phase_frames) % frame_count;
            if age < RIPPLE_FRAMES {
                draw_ripple(pixels, age);
            }
        }
    }
}

fn draw_sparkle(pixels: &mut [[u8; 4]], sparkle: &Sparkle, frame: u32, frame_count: u32) {
    const GLINT: [u8; 4] = [150, 214, 220, 255];
    const SOFT_GLINT: [u8; 4] = [62, 132, 146, 255];
    const SPARKLE_FRAMES: u32 = 24;
    let age = (frame + frame_count - sparkle.phase_frames) % frame_count;
    let [x, y] = sparkle.position;
    if age < SPARKLE_FRAMES {
        let is_peak = (SPARKLE_FRAMES / 3..2 * SPARKLE_FRAMES / 3).contains(&age);
        put(pixels, x, y, if is_peak { GLINT } else { SOFT_GLINT });
        if is_peak {
            for (dx, dy) in [(1, 0), (-1, 0), (0, 1), (0, -1)] {
                put(pixels, x + dx, y + dy, SOFT_GLINT);
            }
        }
    }
}
