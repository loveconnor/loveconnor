//! A wide pond around the koi, built from the rest of the account's activity.
//!
//! The pool floor is the last year's contribution graph as a mosaic. A fry swims for
//! each language, in GitHub's color for it. Pellets are the last seven days'
//! contributions, and the koi eats them as it passes. A lily pad floats for each year
//! on GitHub, and they bloom while a weekly streak runs. Every fish wanders its own path,
//! chosen from the account's ID, and the koi still moves as the day's [`Behavior`] says.
//! Identical inputs produce identical pixels without I/O or
//! clock reads, and every frame count closes the loop seamlessly.

use std::f64::consts::{PI, TAU};

use chrono::{Datelike, Days};

use crate::behavior::{Behavior, select_behavior};
use crate::coat::KoiBody;
use crate::history::{AccountSnapshot, summarize_contributions};
use crate::koi::{KoiDevelopment, KoiDevelopmentError, RecentStreak, derive_koi_development};
use crate::motion::{Course, Meander, MeanderWave, SwimPose, calculate_swim_pose};
use crate::render::{FRAME_HEIGHT_PIXELS, FRAME_WIDTH_PIXELS, render_placed_koi};

/// Width of the native scene; exports scale it up exactly.
pub const POND_WIDTH_PIXELS: u32 = 520;
/// Height of the native scene.
pub const POND_HEIGHT_PIXELS: u32 = 208;

const WIDTH: i32 = POND_WIDTH_PIXELS as i32;
const HEIGHT: i32 = POND_HEIGHT_PIXELS as i32;
/// Three cruise or dart pose loops, and a whole number of every resting pose loop, so
/// each loop is long enough that the paths do not read as a repeating pattern.
const LOOP_FRAMES: u32 = 360;
/// Tries per path; the same account always gets the same choice.
const PATH_CANDIDATES: u64 = 600;
const PATH_SAMPLES: usize = 1_440;

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
    koi_phases: Vec<f64>,
    koi_meander: Meander,
    fry: Vec<Fry>,
    pellets: Vec<Pellet>,
    sparkles: Vec<Sparkle>,
}

struct Fry {
    colors: [[u8; 4]; 3],
    length_pixels: f64,
    /// Each frame's center, heading, turn, and tail swing.
    track: Vec<FryPlace>,
}

#[derive(Clone, Copy)]
struct FryPlace {
    center: [f64; 2],
    forward: [f64; 2],
    turn_radians: f64,
    wag: f64,
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
        let mut fry: Vec<Fry> = Vec::new();
        for (fry_index, (name, share)) in language_shares.iter().enumerate() {
            let color = koi_development
                .language_colors
                .get(*name)
                .copied()
                .flatten()
                .map_or([170, 180, 190], |language_color| language_color.0);
            // Every fourth fry trails the one before it, so small schools form and part.
            let track = if fry_index % 4 == 3 {
                calculate_follower_track(&fry[fry_index - 1].track, fry_index)
            } else {
                calculate_fry_track(seed, fry_index as u64)
            };
            fry.push(Fry {
                colors: shade_fry(color),
                length_pixels: 15.0 + 13.0 * share.sqrt(),
                track,
            });
        }

        let koi_meander = choose_meander(seed, &KOI_PATH);
        let koi_phases = calculate_koi_phases(behavior, &koi_meander, seed);

        let frame_count = LOOP_FRAMES;
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
                    let [x, y] =
                        calculate_mouth_point(&koi_meander, koi_phases[eat_frame as usize]);
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
            koi_phases,
            koi_meander,
            fry,
            pellets,
            sparkles,
        })
    }

    /// Frames in the pond's loop.
    pub fn frame_count(&self) -> u32 {
        LOOP_FRAMES
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
            draw_fry(&mut pixels, &mut swim_mask, fry, frame);
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
            (
                calculate_meander_pose(
                    self.behavior,
                    frame,
                    self.koi_meander,
                    self.koi_phases[frame as usize],
                ),
                [0, 0],
            )
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

fn is_resting(behavior: Behavior) -> bool {
    matches!(behavior, Behavior::Rest | Behavior::DeepRest)
}

/// The range a path may be drawn from and must stay inside.
struct PathLimits {
    /// Allowed head positions: left, top, right, bottom.
    bounds: [f64; 4],
    center_x_range: [f64; 2],
    /// Lowest and highest amplitude of each wave, by axis then wave.
    amplitude_ranges: [[[f64; 2]; 3]; 2],
    /// The tightest turn the fish can make without its body folding.
    min_radius_pixels: f64,
    salt: u64,
}

/// The koi has the length of a body to bring around each turn, so it turns wide.
const KOI_PATH: PathLimits = PathLimits {
    bounds: [78.0, 42.0, 442.0, 166.0],
    center_x_range: [245.0, 275.0],
    amplitude_ranges: [
        [[110.0, 165.0], [10.0, 45.0], [0.0, 20.0]],
        [[18.0, 50.0], [10.0, 38.0], [0.0, 14.0]],
    ],
    min_radius_pixels: 44.0,
    salt: 0x6b6f69,
};

fn fry_path(fry_index: u64) -> PathLimits {
    PathLimits {
        bounds: [24.0, 22.0, 496.0, 186.0],
        center_x_range: [110.0, 410.0],
        amplitude_ranges: [
            [[40.0, 170.0], [10.0, 70.0], [0.0, 30.0]],
            [[15.0, 70.0], [5.0, 45.0], [0.0, 22.0]],
        ],
        min_radius_pixels: 13.0,
        salt: 0x667279 + fry_index,
    }
}

/// Choose the valid candidate that covers the most water, so the same account always
/// swims the same way. A plain oval is the fallback, though none has needed it.
fn choose_meander(seed: u64, limits: &PathLimits) -> Meander {
    let mut best: Option<(f64, Meander)> = None;
    for candidate_index in 0..PATH_CANDIDATES {
        let mut draw_count = 0;
        let mut draw = |low: f64, high: f64| {
            draw_count += 1;
            let mixed = hash(seed ^ limits.salt, candidate_index << 8 | draw_count);
            low + (high - low) * (mixed % 65_536) as f64 / 65_535.0
        };
        let center = [
            draw(limits.center_x_range[0], limits.center_x_range[1]),
            104.0,
        ];
        let mut waves = [[MeanderWave {
            amplitude_pixels: 0.0,
            phase_radians: 0.0,
        }; 3]; 2];
        for axis_index in 0..2 {
            for wave_index in 0..3 {
                let [low, high] = limits.amplitude_ranges[axis_index][wave_index];
                waves[axis_index][wave_index] = MeanderWave {
                    amplitude_pixels: draw(low, high),
                    phase_radians: draw(0.0, TAU),
                };
            }
        }
        let meander = Meander { center, waves };
        if let Some(score) = score_meander(&meander, limits)
            && best
                .as_ref()
                .is_none_or(|(best_score, _)| score > *best_score)
        {
            best = Some((score, meander));
        }
    }
    best.map_or_else(
        || Meander {
            center: [260.0, 104.0],
            waves: [
                [
                    MeanderWave {
                        amplitude_pixels: 150.0,
                        phase_radians: 0.0,
                    },
                    QUIET_WAVE,
                    QUIET_WAVE,
                ],
                [
                    MeanderWave {
                        amplitude_pixels: 50.0,
                        phase_radians: PI / 2.0,
                    },
                    QUIET_WAVE,
                    QUIET_WAVE,
                ],
            ],
        },
        |(_, meander)| meander,
    )
}

const QUIET_WAVE: MeanderWave = MeanderWave {
    amplitude_pixels: 0.0,
    phase_radians: 0.0,
};

/// How much water a path visits and how often it changes which way it is turning, or
/// `None` if it leaves its bounds or turns too tightly.
fn score_meander(meander: &Meander, limits: &PathLimits) -> Option<f64> {
    const COARSE_SAMPLES: usize = 360;
    const VISIT_CELL_PIXELS: f64 = 24.0;
    let [left, top, right, bottom] = limits.bounds;
    let step = TAU / COARSE_SAMPLES as f64;
    let mut visited = std::collections::BTreeSet::new();
    let mut turn_changes = 0;
    let mut previous_turn = 0.0;
    for sample_index in 0..COARSE_SAMPLES {
        let phase = sample_index as f64 * step;
        let [x, y] = meander.calculate_point(phase);
        if x < left || x > right || y < top || y > bottom {
            return None;
        }
        let velocity = meander.calculate_velocity(phase);
        let next_velocity = meander.calculate_velocity(phase + step);
        let speed = velocity[0].hypot(velocity[1]);
        let acceleration = [
            (next_velocity[0] - velocity[0]) / step,
            (next_velocity[1] - velocity[1]) / step,
        ];
        let turn = velocity[0] * acceleration[1] - velocity[1] * acceleration[0];
        if turn * previous_turn < 0.0 {
            turn_changes += 1;
        }
        previous_turn = turn;
        if speed.powi(3) < limits.min_radius_pixels * turn.abs() {
            return None;
        }
        visited.insert((
            (x / VISIT_CELL_PIXELS) as i32,
            (y / VISIT_CELL_PIXELS) as i32,
        ));
    }
    // A bend the other way is worth far more than a little extra water: it is what keeps
    // a path from reading as a lap.
    Some(visited.len() as f64 + 12.0 * f64::from(turn_changes))
}

/// Each sample's share of the way around the path by distance, so a fish can move at the
/// speed it chooses however the waves bunch up.
fn measure_path_shares(meander: &Meander) -> Vec<f64> {
    let mut distances = Vec::with_capacity(PATH_SAMPLES + 1);
    let mut covered = 0.0;
    let mut previous = meander.calculate_point(0.0);
    distances.push(0.0);
    for sample_index in 1..=PATH_SAMPLES {
        let point = meander.calculate_point(TAU * sample_index as f64 / PATH_SAMPLES as f64);
        covered += (point[0] - previous[0]).hypot(point[1] - previous[1]);
        previous = point;
        distances.push(covered);
    }
    distances
        .iter()
        .map(|distance| distance / covered)
        .collect()
}

/// The phase a share of the distance around the path falls at.
fn phase_at_share(path_shares: &[f64], share: f64) -> f64 {
    let whole_laps = share.floor();
    let wanted = share - whole_laps;
    let sample = path_shares
        .partition_point(|covered| *covered < wanted)
        .max(1);
    let (below, above) = (path_shares[sample - 1], path_shares[sample]);
    let within = (wanted - below) / (above - below).max(1e-12);
    TAU * (whole_laps + (sample as f64 - 1.0 + within) / PATH_SAMPLES as f64)
}

/// Turn frame-by-frame speeds into each frame's phase, covering whole laps exactly.
fn phases_from_speeds(
    meander: &Meander,
    speeds: &[f64],
    lap_count: f64,
    start_share: f64,
) -> Vec<f64> {
    let path_shares = measure_path_shares(meander);
    let total_speed: f64 = speeds.iter().sum();
    let mut covered_speed = 0.0;
    speeds
        .iter()
        .map(|speed| {
            let share = start_share + lap_count * covered_speed / total_speed;
            covered_speed += speed;
            phase_at_share(&path_shares, share)
        })
        .collect()
}

/// The koi's phase per frame. Its strokes set the pace, and it also eases up and picks
/// up speed over the loop, as though it had somewhere to be and then did not. A cruise
/// wanders one lap, a dart two.
fn calculate_koi_phases(behavior: Behavior, meander: &Meander, seed: u64) -> Vec<f64> {
    let lap_behavior = calculate_lap_behavior(behavior);
    let lap_angle = |frame: u32| match calculate_swim_pose(lap_behavior, frame).course {
        Course::Lap { angle_radians } => angle_radians,
        _ => unreachable!("lap styles swim laps"),
    };
    let ease_phase = (hash(seed, 0x65617365) % 628) as f64 / 100.0;
    let speeds: Vec<f64> = (0..LOOP_FRAMES)
        .map(|frame| {
            let stroke_speed = (lap_angle(frame + 1) - lap_angle(frame)).rem_euclid(TAU);
            let progress = f64::from(frame) / f64::from(LOOP_FRAMES);
            stroke_speed
                * (1.0
                    + 0.28 * (TAU * 2.0 * progress + ease_phase).sin()
                    + 0.14 * (TAU * 5.0 * progress + ease_phase * 2.0).sin())
        })
        .collect();
    let lap_count = if behavior == Behavior::Dart { 2.0 } else { 1.0 };
    phases_from_speeds(meander, &speeds, lap_count, 0.0)
}

fn calculate_lap_behavior(behavior: Behavior) -> Behavior {
    if behavior == Behavior::Dart {
        Behavior::Dart
    } else {
        Behavior::Cruise
    }
}

/// A pose on the koi's path. Surface feeding swims too: the pond's food comes to it.
fn calculate_meander_pose(
    behavior: Behavior,
    frame: u32,
    meander: Meander,
    phase_radians: f64,
) -> SwimPose {
    SwimPose {
        course: Course::Meander {
            meander,
            phase_radians,
        },
        ..calculate_swim_pose(calculate_lap_behavior(behavior), frame)
    }
}

/// Just ahead of the head along the path, where a pellet meets the mouth.
fn calculate_mouth_point(meander: &Meander, phase_radians: f64) -> [f64; 2] {
    const MOUTH_REACH_PIXELS: f64 = 4.0;
    let [velocity_x, velocity_y] = meander.calculate_velocity(phase_radians);
    meander
        .calculate_point(phase_radians + MOUTH_REACH_PIXELS / velocity_x.hypot(velocity_y).max(1.0))
}

/// A fry wanders its own path in bursts: quick flicks of the tail, then a long glide.
fn calculate_fry_track(seed: u64, fry_index: u64) -> Vec<FryPlace> {
    const TAIL_BEATS_PER_LOOP: f64 = 46.0;
    let meander = choose_meander(seed, &fry_path(fry_index));
    let mixed = hash(seed ^ 0x6275, fry_index);
    let burst_count = 3.0 + (mixed % 5) as f64;
    let burst_phase = ((mixed >> 8) % 628) as f64 / 100.0;
    let start_share = ((mixed >> 20) % 1_000) as f64 / 1_000.0;
    let speeds: Vec<f64> = (0..LOOP_FRAMES)
        .map(|frame| {
            let progress = f64::from(frame) / f64::from(LOOP_FRAMES);
            let burst = (TAU * burst_count * progress + burst_phase).sin().max(0.0);
            0.22 + 1.9 * burst.powi(3)
        })
        .collect();
    let total_speed: f64 = speeds.iter().sum();
    let phases = phases_from_speeds(&meander, &speeds, 1.0, start_share);
    let points: Vec<[f64; 2]> = phases
        .iter()
        .map(|phase| meander.calculate_point(*phase))
        .collect();
    let mut covered_speed = 0.0;
    let wags: Vec<f64> = speeds
        .iter()
        .map(|speed| {
            let wag = (TAU * TAIL_BEATS_PER_LOOP * covered_speed / total_speed).sin();
            covered_speed += speed;
            wag
        })
        .collect();
    (0..LOOP_FRAMES as usize)
        .map(|frame| {
            let forward = normalize(meander.calculate_velocity(phases[frame]));
            FryPlace {
                center: points[frame],
                forward,
                turn_radians: 0.0,
                wag: wags[frame],
            }
        })
        .collect::<Vec<_>>()
        .with_turns()
}

/// A fry that follows another a little behind and to one side, swaying on its own.
fn calculate_follower_track(leader: &[FryPlace], fry_index: usize) -> Vec<FryPlace> {
    let frame_total = leader.len();
    let lag_frames = 16 + fry_index % 3 * 6;
    let side_pixels = if fry_index % 8 == 3 { 9.0 } else { -11.0 };
    (0..frame_total)
        .map(|frame| {
            let place = leader[(frame + frame_total - lag_frames) % frame_total];
            let progress = frame as f64 / frame_total as f64;
            let sway = side_pixels + 4.0 * (TAU * 7.0 * progress).sin();
            FryPlace {
                center: [
                    place.center[0] - place.forward[1] * sway,
                    place.center[1] + place.forward[0] * sway,
                ],
                ..place
            }
        })
        .collect()
}

trait WithTurns {
    fn with_turns(self) -> Vec<FryPlace>;
}

impl WithTurns for Vec<FryPlace> {
    /// Fill in how sharply each fry is turning, from its heading a moment ago, so the
    /// body curves into a turn instead of pivoting like a compass needle.
    fn with_turns(mut self) -> Vec<FryPlace> {
        const TURN_LOOKBACK_FRAMES: usize = 7;
        let frame_total = self.len();
        let headings: Vec<f64> = self
            .iter()
            .map(|place| place.forward[1].atan2(place.forward[0]))
            .collect();
        for frame in 0..frame_total {
            let earlier = headings[(frame + frame_total - TURN_LOOKBACK_FRAMES) % frame_total];
            self[frame].turn_radians = (headings[frame] - earlier + PI).rem_euclid(TAU) - PI;
        }
        self
    }
}

fn normalize([x, y]: [f64; 2]) -> [f64; 2] {
    let length = x.hypot(y).max(1e-9);
    [x / length, y / length]
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

fn draw_fry(pixels: &mut [[u8; 4]], swim_mask: &mut [bool], fry: &Fry, frame: u32) {
    let place = fry.track[frame as usize];
    let (center, forward) = (place.center, place.forward);
    let wag = place.wag;
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
            let bend_share = (along / (length + tail_length)).clamp(0.0, 1.0).powi(2);
            let bend = bend_share * (wag * 1.8 - place.turn_radians * length * 0.9);
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
