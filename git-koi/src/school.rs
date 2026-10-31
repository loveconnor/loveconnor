//! How the pond's fish behave: a simulation of each one deciding, frame by frame, where
//! to go.
//!
//! Fry swim in tail beats and glides, rest, form loose schools, keep their distance from
//! each other, and scatter from the koi. Food lands in small sprinkles at random; any fish
//! that notices it heads over, and the first mouth to arrive eats it. Near the end of the
//! loop each fish makes its way back to where it started, so the loop closes seamlessly.

use std::f64::consts::{PI, TAU};

const POND_BOUNDS: [f64; 4] = [22.0, 22.0, 498.0, 186.0];
/// The whole fish, tail included, stays inside these: the shore's inner edge.
const WATER_BOUNDS: [f64; 4] = [9.0, 9.0, 511.0, 199.0];
/// Over these last frames a fry goes straight home, still making room for others, so
/// nothing has to be pulled into place afterward.
const ARRIVAL_FRAMES: f64 = 80.0;
/// The koi turns back before its long body could reach the shore.
const KOI_BOUNDS: [f64; 4] = [84.0, 62.0, 436.0, 146.0];
const KOI_TURN_RADIUS_PIXELS: f64 = 34.0;
const KOI_CRUISE_SPEED: f64 = 2.1;
const KOI_DART_SPEED: f64 = 3.4;
const KOI_SIGHT_PIXELS: f64 = 260.0;
const KOI_BITE_PIXELS: f64 = 13.0;
const FRY_SIGHT_PIXELS: f64 = 170.0;
const FRY_BITE_PIXELS: f64 = 4.5;
const FRY_MAX_SPEED: f64 = 3.4;
/// However hemmed in a fry is, it can only move this far in a frame: it swims out of the
/// way rather than being shoved.
const FRY_MAX_STEP_PIXELS: f64 = 4.2;
const SPRINKLE_MAX_PELLETS: u32 = 5;
/// Share of the loop over which food may land, and by which leftovers have sunk.
const LANDING_SHARE: f64 = 0.62;
const SINK_SHARE: f64 = 0.86;
/// Frames over which any small miss at the end of the loop is smoothed away.
const CLOSING_FRAMES: usize = 90;
/// The stretches of the koi's body, as shares of its length and of a fixed width.
const KOI_BODY_CIRCLES: [(f64, f64); 5] = [
    (0.08, 22.0),
    (0.3, 24.0),
    (0.52, 18.0),
    (0.76, 11.0),
    (1.08, 19.0),
];

/// What the simulation needs to know about the pond's fish and food.
pub(crate) struct SchoolSetup {
    pub seed: u64,
    pub frame_count: usize,
    pub fry_lengths: Vec<f64>,
    pub pellet_count: u32,
    pub koi_body_length: f64,
    pub koi_motion: KoiMotion,
}

/// How the koi moves today.
pub(crate) enum KoiMotion {
    /// Swims, at a speed multiplied per frame by its stroke's surge.
    Swimming { dart: bool, surges: Vec<f64> },
    /// Stays still, with its head here and its body trailing along this angle.
    Hovering { head: [f64; 2], backward_angle: f64 },
}

/// One fry in one frame.
#[derive(Clone, Copy)]
pub(crate) struct FryPlace {
    pub head: [f64; 2],
    pub forward: [f64; 2],
    pub turn_radians: f64,
    pub wag: f64,
}

/// One pellet's life.
#[derive(Clone, Copy)]
pub(crate) struct PelletLife {
    pub position: [f64; 2],
    pub landed_at: u32,
    pub gone_at: u32,
    pub eaten_by_koi: bool,
    pub sank: bool,
}

/// Everything that happens in the pond over one loop.
pub(crate) struct School {
    pub koi_heads: Vec<[f64; 2]>,
    pub fry: Vec<Vec<FryPlace>>,
    pub pellets: Vec<PelletLife>,
}

/// A small, stable random source, so the same inputs always swim the same way.
struct Dice(u64);

impl Dice {
    fn roll(&mut self) -> f64 {
        self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut mixed = self.0;
        mixed = (mixed ^ (mixed >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        mixed = (mixed ^ (mixed >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        ((mixed ^ (mixed >> 31)) >> 11) as f64 / (1_u64 << 53) as f64
    }

    fn between(&mut self, low: f64, high: f64) -> f64 {
        low + (high - low) * self.roll()
    }

    fn point_within(&mut self, [left, top, right, bottom]: [f64; 4]) -> [f64; 2] {
        [self.between(left, right), self.between(top, bottom)]
    }
}

struct Fish {
    head: [f64; 2],
    heading: f64,
    speed: f64,
    length: f64,
    goal: [f64; 2],
    rest_frames: u32,
    beat_frames: u32,
    until_next_beat: u32,
    tail_phase: f64,
    start_head: [f64; 2],
    start_heading: f64,
    is_going_home: bool,
    /// Frames until this fry is hungry again after eating.
    full_frames: u32,
}

impl Fish {
    fn forward(&self) -> [f64; 2] {
        [self.heading.cos(), self.heading.sin()]
    }

    /// The middle of the fish, tail included, and how far it reaches from there.
    fn circle(&self) -> ([f64; 2], f64) {
        let [forward_x, forward_y] = self.forward();
        (
            [
                self.head[0] - forward_x * self.length * 0.7,
                self.head[1] - forward_y * self.length * 0.7,
            ],
            self.length * 0.72,
        )
    }
}

fn distance(from: [f64; 2], to: [f64; 2]) -> f64 {
    (to[0] - from[0]).hypot(to[1] - from[1])
}

fn angle_between(from: f64, to: f64) -> f64 {
    (to - from + PI).rem_euclid(TAU) - PI
}

/// A push away from walls that strengthens toward the edge.
fn wall_push(point: [f64; 2], [left, top, right, bottom]: [f64; 4], margin: f64) -> [f64; 2] {
    let inward = |near: f64, far: f64| {
        (1.0 - near / margin).clamp(0.0, 1.0) - (1.0 - far / margin).clamp(0.0, 1.0)
    };
    [
        inward(point[0] - left, right - point[0]),
        inward(point[1] - top, bottom - point[1]),
    ]
}

/// Swim the loop several ways and keep the best: the koi's whole body stays in the water
/// and it gets a fair share of the food. The choice is as repeatable as each swim.
pub(crate) fn simulate_school(setup: &SchoolSetup) -> School {
    const ATTEMPTS: u64 = 60;
    let wanted_bites = setup.pellet_count as usize / 4;
    let mut best: Option<((usize, usize), School)> = None;
    for attempt in 0..ATTEMPTS {
        let school = swim_once(
            setup,
            setup.seed.wrapping_add(attempt.wrapping_mul(0x1_0001)),
        );
        let koi_bites = school
            .pellets
            .iter()
            .filter(|pellet| pellet.eaten_by_koi)
            .count();
        let sunk = school.pellets.iter().filter(|pellet| pellet.sank).count();
        let fry_bites = school.pellets.len() - koi_bites - sunk;
        let score = (
            count_frames_out_of_water(setup, &school) + count_frames_touching(setup, &school),
            wanted_bites.saturating_sub(koi_bites) + wanted_bites.saturating_sub(fry_bites) + sunk,
        );
        if best
            .as_ref()
            .is_none_or(|(best_score, _)| score < *best_score)
        {
            best = Some((score, school));
        }
    }
    best.expect("at least one attempt ran").1
}

/// Frames in which two fry touch, or a fry is against the koi's body.
fn count_frames_touching(setup: &SchoolSetup, school: &School) -> usize {
    /// How near the koi's spine the middle of a fry may come.
    const KOI_CLEARANCE_PIXELS: f64 = 24.0;
    let frame_total = setup.frame_count;
    let is_swimming = matches!(setup.koi_motion, KoiMotion::Swimming { .. });
    (0..frame_total)
        .filter(|frame| {
            let circles: Vec<([f64; 2], f64)> = school
                .fry
                .iter()
                .zip(&setup.fry_lengths)
                .map(|(track, length)| {
                    let place = track[*frame];
                    (
                        [
                            place.head[0] - place.forward[0] * length * 0.7,
                            place.head[1] - place.forward[1] * length * 0.7,
                        ],
                        length * 0.6,
                    )
                })
                .collect();
            let fry_touch = circles.iter().enumerate().any(|(index, first)| {
                circles[index + 1..]
                    .iter()
                    .any(|second| distance(first.0, second.0) < first.1 + second.1)
            });
            if fry_touch || !is_swimming {
                return fry_touch;
            }
            // Read the koi's body back along its trail, a few pixels at a time.
            let mut covered = 0.0;
            let mut point = school.koi_heads[*frame];
            for step_back in 1..frame_total {
                let earlier = school.koi_heads[(frame + frame_total - step_back) % frame_total];
                covered += distance(point, earlier);
                point = earlier;
                if circles
                    .iter()
                    .any(|(center, _)| distance(*center, point) < KOI_CLEARANCE_PIXELS)
                {
                    return true;
                }
                if covered >= setup.koi_body_length {
                    break;
                }
            }
            false
        })
        .count()
}

/// Frames in which any of the koi, fins and tail included, could reach the shore.
fn count_frames_out_of_water(setup: &SchoolSetup, school: &School) -> usize {
    /// Share of body length along the trail, and how far the fish reaches from there.
    const KOI_REACHES: [(f64, f64); 5] = [
        (0.0, 12.0),
        (0.25, 36.0),
        (0.55, 24.0),
        (0.9, 14.0),
        (1.0, 38.0),
    ];
    const SHORE_PIXELS: f64 = 6.0;
    if matches!(setup.koi_motion, KoiMotion::Hovering { .. }) {
        return 0;
    }
    let heads = &school.koi_heads;
    let frame_total = heads.len();
    (0..frame_total)
        .filter(|frame| {
            KOI_REACHES.iter().any(|(share, reach)| {
                let wanted = share * setup.koi_body_length;
                let mut covered = 0.0;
                let mut point = heads[*frame];
                for step_back in 1..frame_total {
                    let earlier = heads[(frame + frame_total - step_back) % frame_total];
                    covered += distance(point, earlier);
                    point = earlier;
                    if covered >= wanted {
                        break;
                    }
                }
                point[0] - reach < SHORE_PIXELS
                    || point[1] - reach < SHORE_PIXELS
                    || point[0] + reach > 520.0 - SHORE_PIXELS
                    || point[1] + reach > 208.0 - SHORE_PIXELS
            })
        })
        .count()
}

/// Play a school at a higher frame rate: positions follow a smooth curve through the
/// simulated frames, so nothing moves in straight segments between them.
pub(crate) fn resample_school(school: School, shown_frames: usize) -> School {
    let simulated_frames = school.koi_heads.len();
    let time_of = |shown: usize| shown as f64 * simulated_frames as f64 / shown_frames as f64;
    let koi_heads = (0..shown_frames)
        .map(|shown| sample_curve(&school.koi_heads, time_of(shown)))
        .collect();
    let fry = school
        .fry
        .iter()
        .map(|track| {
            let heads: Vec<[f64; 2]> = track.iter().map(|place| place.head).collect();
            (0..shown_frames)
                .map(|shown| {
                    let time = time_of(shown);
                    let earlier = track[time.floor() as usize % simulated_frames];
                    let later = track[(time.floor() as usize + 1) % simulated_frames];
                    let share = time - time.floor();
                    let mix = |from: f64, to: f64| from + (to - from) * share;
                    FryPlace {
                        head: sample_curve(&heads, time),
                        forward: normalize([
                            mix(earlier.forward[0], later.forward[0]),
                            mix(earlier.forward[1], later.forward[1]),
                        ]),
                        turn_radians: mix(earlier.turn_radians, later.turn_radians),
                        wag: mix(earlier.wag, later.wag),
                    }
                })
                .collect()
        })
        .collect();
    let scale_frame = |frame: u32| {
        (f64::from(frame) * shown_frames as f64 / simulated_frames as f64).round() as u32
    };
    let pellets = school
        .pellets
        .iter()
        .map(|pellet| PelletLife {
            landed_at: scale_frame(pellet.landed_at),
            gone_at: scale_frame(pellet.gone_at),
            ..*pellet
        })
        .collect();
    School {
        koi_heads,
        fry,
        pellets,
    }
}

/// A Catmull-Rom curve through a looping track, at a moment between its frames.
fn sample_curve(points: &[[f64; 2]], time: f64) -> [f64; 2] {
    let total = points.len();
    let whole = time.floor() as usize;
    let share = time - time.floor();
    let at = |offset: usize| points[(whole + total + offset - 1) % total];
    let (before, here, next, after) = (at(0), at(1), at(2), at(3));
    [0, 1].map(|axis| {
        let (p0, p1, p2, p3) = (before[axis], here[axis], next[axis], after[axis]);
        0.5 * (2.0 * p1
            + (p2 - p0) * share
            + (2.0 * p0 - 5.0 * p1 + 4.0 * p2 - p3) * share * share
            + (3.0 * p1 - p0 - 3.0 * p2 + p3) * share * share * share)
    })
}

/// Simulate one whole loop and close it.
fn swim_once(setup: &SchoolSetup, seed: u64) -> School {
    let frame_count = setup.frame_count;
    let mut dice = Dice(seed ^ 0x706f_6e64);
    let pellets_planned = plan_pellets(setup, &mut dice);
    let mut pellets = pellets_planned;
    let sink_frame = (frame_count as f64 * SINK_SHARE) as u32;

    let koi_start_heading = dice.between(-0.5, 0.5);
    let koi_start_head = [dice.between(200.0, 320.0), dice.between(60.0, 148.0)];
    let mut koi_head = koi_start_head;
    let mut koi_heading = koi_start_heading;
    let mut koi_goal = dice.point_within(KOI_BOUNDS);
    let mut koi_heads: Vec<[f64; 2]> = Vec::with_capacity(frame_count + 1);
    let mut koi_going_home = false;

    let mut fish: Vec<Fish> = setup
        .fry_lengths
        .iter()
        .map(|length| {
            let head = dice.point_within(POND_BOUNDS);
            let heading = dice.between(0.0, TAU);
            Fish {
                head,
                heading,
                speed: 0.3,
                length: *length,
                goal: dice.point_within(POND_BOUNDS),
                rest_frames: 0,
                beat_frames: 0,
                until_next_beat: dice.between(4.0, 40.0) as u32,
                tail_phase: 0.0,
                start_head: head,
                start_heading: heading,
                is_going_home: false,
                full_frames: dice.between(0.0, 60.0) as u32,
            }
        })
        .collect();
    let mut koi_circles = koi_body_circles(setup, &koi_heads, koi_head, koi_heading);
    for _ in 0..60 {
        separate(&mut fish, &koi_circles);
    }
    for one_fish in &mut fish {
        one_fish.start_head = one_fish.head;
    }

    let mut fry_records: Vec<Vec<FryPlace>> = vec![Vec::with_capacity(frame_count + 1); fish.len()];
    for frame in 0..=frame_count {
        koi_heads.push(koi_head);
        for (one_fish, record) in fish.iter().zip(&mut fry_records) {
            record.push(FryPlace {
                head: one_fish.head,
                forward: one_fish.forward(),
                turn_radians: 0.0,
                wag: one_fish.tail_phase,
            });
        }
        if frame == frame_count {
            break;
        }
        let frames_left = (frame_count - frame) as f64;
        let frame_number = frame as u32;
        let is_afloat =
            |pellet: &PelletLife| pellet.landed_at <= frame_number && frame_number < pellet.gone_at;

        if let KoiMotion::Swimming { dart, surges } = &setup.koi_motion {
            let base_speed = if *dart {
                KOI_DART_SPEED
            } else {
                KOI_CRUISE_SPEED
            };
            if !koi_going_home
                && distance(koi_head, koi_start_head) / base_speed > frames_left * 0.72
            {
                koi_going_home = true;
            }
            let forward = [koi_heading.cos(), koi_heading.sin()];
            let mut speed = base_speed * surges[frame % surges.len()];
            let steer_to = if koi_going_home {
                let lead_distance = 46.0_f64.min(distance(koi_head, koi_start_head) * 0.6);
                let lead = [
                    koi_start_head[0] - koi_start_heading.cos() * lead_distance,
                    koi_start_head[1] - koi_start_heading.sin() * lead_distance,
                ];
                speed = (distance(koi_head, koi_start_head) / frames_left.max(1.0))
                    .clamp(base_speed * 0.6, base_speed * 1.3)
                    * (surges[frame % surges.len()] * 0.5 + 0.5);
                lead
            } else {
                let reachable_food = pellets
                    .iter()
                    .filter(|pellet| is_afloat(pellet))
                    // Food by the shore is for the fry; the koi cannot turn there.
                    .filter(|pellet| {
                        pellet.position[0] > KOI_BOUNDS[0]
                            && pellet.position[0] < KOI_BOUNDS[2]
                            && pellet.position[1] > KOI_BOUNDS[1]
                            && pellet.position[1] < KOI_BOUNDS[3]
                    })
                    .filter(|pellet| {
                        let offset = distance(koi_head, pellet.position);
                        let bearing = (pellet.position[1] - koi_head[1])
                            .atan2(pellet.position[0] - koi_head[0]);
                        let off_nose = angle_between(koi_heading, bearing).abs();
                        // Food inside its turning circle is out of reach for now.
                        offset < KOI_SIGHT_PIXELS
                            && offset > 2.0 * KOI_TURN_RADIUS_PIXELS * off_nose.sin().abs()
                            && off_nose < 2.2
                    })
                    .min_by(|left, right| {
                        distance(koi_head, left.position)
                            .total_cmp(&distance(koi_head, right.position))
                    });
                if let Some(pellet) = reachable_food {
                    speed *= 1.3;
                    pellet.position
                } else {
                    if distance(koi_head, koi_goal) < 45.0 {
                        // Somewhere it isn't: fish do not dither over a few body lengths.
                        for _ in 0..12 {
                            koi_goal = dice.point_within(KOI_BOUNDS);
                            if distance(koi_head, koi_goal) > 170.0 {
                                break;
                            }
                        }
                    }
                    koi_goal
                }
            };
            let look_ahead = [
                koi_head[0] + forward[0] * 54.0,
                koi_head[1] + forward[1] * 54.0,
            ];
            let push = wall_push(look_ahead, KOI_BOUNDS, 20.0);
            let toward = normalize([steer_to[0] - koi_head[0], steer_to[1] - koi_head[1]]);
            let desired = [toward[0] + push[0] * 4.0, toward[1] + push[1] * 4.0];
            let turn_limit = speed / KOI_TURN_RADIUS_PIXELS;
            koi_heading += angle_between(koi_heading, desired[1].atan2(desired[0]))
                .clamp(-turn_limit, turn_limit);
            koi_head[0] += koi_heading.cos() * speed;
            koi_head[1] += koi_heading.sin() * speed;
        }
        koi_circles = koi_body_circles(setup, &koi_heads, koi_head, koi_heading);

        let snapshot: Vec<([f64; 2], [f64; 2], f64)> = fish
            .iter()
            .map(|one_fish| {
                let (center, reach) = one_fish.circle();
                (center, one_fish.forward(), reach)
            })
            .collect();
        let previous_heads: Vec<[f64; 2]> = fish.iter().map(|one_fish| one_fish.head).collect();
        for (fish_index, one_fish) in fish.iter_mut().enumerate() {
            step_fry(
                one_fish,
                fish_index,
                &snapshot,
                &koi_circles,
                &pellets,
                frame_number,
                frames_left,
                &mut dice,
            );
        }
        // Many gentle passes, each limited to what a fry could really swim in a frame, so
        // room spreads through a crowd instead of one fish being shoved through another.
        for _ in 0..8 {
            separate(&mut fish, &koi_circles);
            for (one_fish, before) in fish.iter_mut().zip(&previous_heads) {
                let moved = distance(*before, one_fish.head);
                if moved > FRY_MAX_STEP_PIXELS {
                    let keep = FRY_MAX_STEP_PIXELS / moved;
                    one_fish.head = [
                        before[0] + (one_fish.head[0] - before[0]) * keep,
                        before[1] + (one_fish.head[1] - before[1]) * keep,
                    ];
                }
            }
        }

        // Mouths: the koi's first, since it arrives with more of a rush.
        let mouth = [
            koi_head[0] + koi_heading.cos() * 3.0,
            koi_head[1] + koi_heading.sin() * 3.0,
        ];
        let koi_can_eat = matches!(setup.koi_motion, KoiMotion::Swimming { .. });
        for pellet in pellets.iter_mut() {
            if !(pellet.landed_at <= frame_number && frame_number < pellet.gone_at) {
                continue;
            }
            if koi_can_eat && distance(mouth, pellet.position) < KOI_BITE_PIXELS {
                pellet.gone_at = frame_number + 1;
                pellet.eaten_by_koi = true;
                pellet.sank = false;
            } else if let Some(eater) = fish.iter_mut().find(|one_fish| {
                one_fish.full_frames == 0
                    && !one_fish.is_going_home
                    && distance(one_fish.head, pellet.position) < FRY_BITE_PIXELS
            }) {
                pellet.gone_at = frame_number + 1;
                pellet.sank = false;
                eater.full_frames = dice.between(50.0, 120.0) as u32;
            }
        }
    }
    for pellet in &mut pellets {
        pellet.gone_at = pellet.gone_at.min(sink_frame);
    }

    close_loop_points(&mut koi_heads);
    let fry = fry_records
        .into_iter()
        .map(|mut record| {
            close_fry(&mut record);
            record
        })
        .collect();
    School {
        koi_heads,
        fry,
        pellets,
    }
}

fn normalize([x, y]: [f64; 2]) -> [f64; 2] {
    let length = x.hypot(y).max(1e-9);
    [x / length, y / length]
}

/// Food lands in sprinkles: a few pellets close together in a few moments, somewhere
/// away from the last sprinkle, at random times.
fn plan_pellets(setup: &SchoolSetup, dice: &mut Dice) -> Vec<PelletLife> {
    let frame_count = setup.frame_count as f64;
    let sink_frame = (frame_count * SINK_SHARE) as u32;
    let mut pellets = Vec::new();
    let mut remaining = setup.pellet_count;
    let sprinkle_count = setup.pellet_count.div_ceil(SPRINKLE_MAX_PELLETS).max(1);
    let spacing = frame_count * LANDING_SHARE / f64::from(sprinkle_count);
    let mut last_spot = [260.0, 104.0];
    let mut sprinkle_index = 0;
    while remaining > 0 {
        let pellets_here =
            (dice.between(2.0, f64::from(SPRINKLE_MAX_PELLETS) + 1.0) as u32).min(remaining);
        remaining -= pellets_here;
        let mut spot = dice.point_within([70.0, 50.0, 450.0, 158.0]);
        for _ in 0..20 {
            if distance(spot, last_spot) > 130.0 {
                break;
            }
            spot = dice.point_within([70.0, 50.0, 450.0, 158.0]);
        }
        last_spot = spot;
        let start = spacing * (f64::from(sprinkle_index) + dice.between(0.05, 0.75)) + 15.0;
        for _ in 0..pellets_here {
            let landed_at = (start + dice.between(0.0, 22.0)) as u32;
            // Food left alone gets waterlogged and sinks, each pellet in its own time.
            let sinks_at = landed_at + dice.between(150.0, 260.0) as u32;
            pellets.push(PelletLife {
                position: [
                    (spot[0] + dice.between(-22.0, 22.0)).clamp(30.0, 490.0),
                    (spot[1] + dice.between(-14.0, 14.0)).clamp(28.0, 180.0),
                ],
                landed_at,
                gone_at: sinks_at.min(sink_frame),
                eaten_by_koi: false,
                sank: true,
            });
        }
        sprinkle_index += 1;
    }
    pellets
}

/// The circles covering the koi's body and tail, read back along where its head has been.
fn koi_body_circles(
    setup: &SchoolSetup,
    koi_heads: &[[f64; 2]],
    koi_head: [f64; 2],
    koi_heading: f64,
) -> Vec<([f64; 2], f64)> {
    let length = setup.koi_body_length;
    if let KoiMotion::Hovering {
        head,
        backward_angle,
    } = setup.koi_motion
    {
        return KOI_BODY_CIRCLES
            .iter()
            .map(|(share, radius)| {
                (
                    [
                        head[0] + backward_angle.cos() * share * length,
                        head[1] + backward_angle.sin() * share * length,
                    ],
                    *radius,
                )
            })
            .collect();
    }
    KOI_BODY_CIRCLES
        .iter()
        .map(|(share, radius)| {
            let wanted = share * length;
            let mut covered = 0.0;
            let mut point = koi_head;
            for earlier in koi_heads.iter().rev() {
                let step = distance(point, *earlier);
                if covered + step >= wanted {
                    let within = (wanted - covered) / step.max(1e-9);
                    return (
                        [
                            point[0] + (earlier[0] - point[0]) * within,
                            point[1] + (earlier[1] - point[1]) * within,
                        ],
                        *radius,
                    );
                }
                covered += step;
                point = *earlier;
            }
            // Before the loop began, the koi had been swimming straight.
            let left = wanted - covered;
            (
                [
                    point[0] - koi_heading.cos() * left,
                    point[1] - koi_heading.sin() * left,
                ],
                *radius,
            )
        })
        .collect()
}

/// Push apart any fish that have come too close; the koi does not yield.
fn separate(fish: &mut [Fish], koi_circles: &[([f64; 2], f64)]) {
    const CLEARANCE_PIXELS: f64 = 2.0;
    for first in 0..fish.len() {
        for second in first + 1..fish.len() {
            let (first_center, first_reach) = fish[first].circle();
            let (second_center, second_reach) = fish[second].circle();
            let gap = distance(first_center, second_center);
            let wanted = first_reach + second_reach + CLEARANCE_PIXELS;
            if gap < wanted {
                let away = normalize([
                    second_center[0] - first_center[0],
                    second_center[1] - first_center[1],
                ]);
                let push = (wanted - gap) / 2.0;
                for axis in 0..2 {
                    fish[first].head[axis] -= away[axis] * push;
                    fish[second].head[axis] += away[axis] * push;
                }
            }
        }
    }
    for one_fish in fish.iter_mut() {
        for (circle_center, circle_radius) in koi_circles {
            let (center, reach) = one_fish.circle();
            let gap = distance(*circle_center, center);
            let wanted = circle_radius + reach + CLEARANCE_PIXELS;
            if gap < wanted {
                let away = normalize([center[0] - circle_center[0], center[1] - circle_center[1]]);
                for axis in 0..2 {
                    one_fish.head[axis] += away[axis] * (wanted - gap);
                }
            }
        }
        // Keep the whole fish in the water, tail and all.
        let (center, reach) = one_fish.circle();
        for (axis, (low, high)) in [
            (WATER_BOUNDS[0], WATER_BOUNDS[2]),
            (WATER_BOUNDS[1], WATER_BOUNDS[3]),
        ]
        .into_iter()
        .enumerate()
        {
            let overshoot =
                (low - (center[axis] - reach)).max(0.0) - ((center[axis] + reach) - high).max(0.0);
            one_fish.head[axis] += overshoot;
        }
    }
}

/// One fry's decision for one frame.
#[allow(clippy::too_many_arguments)]
fn step_fry(
    fish: &mut Fish,
    fish_index: usize,
    others: &[([f64; 2], [f64; 2], f64)],
    koi_circles: &[([f64; 2], f64)],
    pellets: &[PelletLife],
    frame: u32,
    frames_left: f64,
    dice: &mut Dice,
) {
    const AVOID_RANGE_PIXELS: f64 = 14.0;
    const KOI_ALARM_PIXELS: f64 = 42.0;
    let (center, reach) = fish.circle();
    let home_distance = distance(fish.head, fish.start_head);
    if !fish.is_going_home && home_distance / 1.3 > frames_left * 0.6 {
        fish.is_going_home = true;
    }
    // Fry look for food until it is time to head home.
    let food = pellets
        .iter()
        .filter(|pellet| pellet.landed_at <= frame && frame < pellet.gone_at)
        .map(|pellet| (distance(fish.head, pellet.position), pellet.position))
        .filter(|(reach_to, _)| *reach_to < FRY_SIGHT_PIXELS)
        // A fry leaves food alone when the koi is nearer to it: it is not worth the risk.
        .filter(|(reach_to, position)| {
            koi_circles.first().is_none_or(|(koi_front, _)| {
                distance(*koi_front, *position) > reach_to * 0.6 + 20.0
            })
        })
        .min_by(|left, right| left.0.total_cmp(&right.0));
    fish.full_frames = fish.full_frames.saturating_sub(1);
    let food = food.filter(|_| fish.full_frames == 0);
    let is_chasing = food.is_some() && !fish.is_going_home;
    if is_chasing {
        fish.rest_frames = 0;
    }
    let mut goal = if fish.is_going_home {
        let lead = home_distance.min(14.0);
        [
            fish.start_head[0] - fish.start_heading.cos() * lead,
            fish.start_head[1] - fish.start_heading.sin() * lead,
        ]
    } else if let Some((_, position)) = food {
        position
    } else {
        fish.goal
    };
    if !fish.is_going_home && !is_chasing && distance(fish.head, fish.goal) < 16.0 {
        fish.goal = dice.point_within(POND_BOUNDS);
        goal = fish.goal;
        if dice.roll() < 0.55 {
            fish.rest_frames = dice.between(24.0, 110.0) as u32;
        }
    }
    let to_goal = normalize([goal[0] - fish.head[0], goal[1] - fish.head[1]]);
    let mut desired = to_goal;
    let mut alarm = 0.0_f64;
    for (other_index, (other_center, other_forward, other_reach)) in others.iter().enumerate() {
        if other_index == fish_index {
            continue;
        }
        let gap = distance(center, *other_center);
        let range = reach + other_reach + AVOID_RANGE_PIXELS;
        if gap < range {
            let away = normalize([center[0] - other_center[0], center[1] - other_center[1]]);
            let strength = 3.2 * (1.0 - gap / range);
            desired[0] += away[0] * strength;
            desired[1] += away[1] * strength;
        }
        // Neighbors of a school swim roughly the same way, at a comfortable distance.
        if other_index / 3 == fish_index / 3 && gap < 90.0 && !is_chasing && !fish.is_going_home {
            desired[0] +=
                other_forward[0] * 0.35 + (other_center[0] - center[0]) / gap.max(1.0) * 0.2;
            desired[1] +=
                other_forward[1] * 0.35 + (other_center[1] - center[1]) / gap.max(1.0) * 0.2;
        }
    }
    for (circle_center, circle_radius) in koi_circles {
        let gap = distance(center, *circle_center);
        let range = circle_radius + reach + KOI_ALARM_PIXELS;
        if gap < range {
            let away = normalize([center[0] - circle_center[0], center[1] - circle_center[1]]);
            let strength = 7.0 * (1.0 - gap / range);
            desired[0] += away[0] * strength;
            desired[1] += away[1] * strength;
            alarm = alarm.max(1.0 - gap / range);
        }
    }
    let push = wall_push(fish.head, POND_BOUNDS, 20.0);
    desired[0] += push[0] * 3.0;
    desired[1] += push[1] * 3.0;

    let is_resting = fish.rest_frames > 0 && alarm == 0.0 && !fish.is_going_home;
    if is_resting {
        fish.rest_frames -= 1;
    }
    // A beat of the tail adds speed; between beats the fish coasts and slows.
    fish.speed *= 0.965;
    if fish.beat_frames > 0 {
        fish.beat_frames -= 1;
        fish.speed = (fish.speed + 0.21).min(FRY_MAX_SPEED);
    } else if fish.until_next_beat == 0 {
        if !is_resting {
            fish.beat_frames = dice.between(4.0, 8.0) as u32;
        }
        let urgency = if alarm > 0.0 || is_chasing { 0.3 } else { 1.0 };
        fish.until_next_beat = (dice.between(16.0, 62.0) * urgency) as u32;
    } else {
        fish.until_next_beat -= 1;
    }
    if alarm > 0.12 {
        fish.beat_frames = fish.beat_frames.max(3);
    }
    if fish.is_going_home {
        // Arrive on time and unhurried, rather than racing there and waiting.
        let wanted_speed =
            (home_distance / (frames_left - 12.0).max(1.0)).clamp(0.05, FRY_MAX_SPEED);
        fish.speed = fish.speed.min(wanted_speed * 1.8);
        if fish.speed < wanted_speed {
            fish.speed = wanted_speed;
        }
    }
    if frames_left <= ARRIVAL_FRAMES {
        // The final approach: exactly the pace that arrives on the last frame.
        let step = home_distance / frames_left;
        let away_from_home = [
            fish.start_head[0] - fish.head[0],
            fish.start_head[1] - fish.head[1],
        ];
        let toward_home = normalize(away_from_home);
        fish.head[0] += toward_home[0] * step;
        fish.head[1] += toward_home[1] * step;
        fish.speed = step;
        // Face the way it is going, then settle into the pose the loop starts in, turning
        // by exactly enough to finish on the last frame.
        if frames_left > 36.0 && step > 0.25 {
            let turn = angle_between(fish.heading, toward_home[1].atan2(toward_home[0]));
            fish.heading += turn.clamp(-0.12, 0.12);
        } else if frames_left <= 36.0 {
            fish.heading += angle_between(fish.heading, fish.start_heading) / frames_left;
        }
        fish.tail_phase += 0.1 + step * 0.5;
        return;
    }
    let wanted_heading = desired[1].atan2(desired[0]);
    let agility = ((fish.speed + 0.2) / 1.0).clamp(0.12, 1.0);
    let turn_limit = 0.17 * agility;
    let turn = angle_between(fish.heading, wanted_heading).clamp(-turn_limit, turn_limit);
    let turn = if is_resting { turn * 0.25 } else { turn };
    fish.heading += turn;
    fish.head[0] += fish.heading.cos() * fish.speed;
    fish.head[1] += fish.heading.sin() * fish.speed;
    fish.tail_phase += 0.1 + fish.speed * 0.5 + if fish.beat_frames > 0 { 0.25 } else { 0.0 };
}

/// The smooth correction that carries a track from wherever it ended to exactly where
/// it began, matching its velocity too, so nothing jumps where the loop restarts.
fn closing_correction(
    frames_total: usize,
    end_gap: [f64; 2],
    end_velocity_gap: [f64; 2],
) -> Vec<[f64; 2]> {
    let window = CLOSING_FRAMES.min(frames_total - 1);
    let mut corrections = vec![[0.0; 2]; frames_total + 1];
    for frame in frames_total - window..=frames_total {
        let share = (frame + window - frames_total) as f64 / window as f64;
        let blend = share * share * (3.0 - 2.0 * share);
        let velocity_blend = share * share * (share - 1.0) * window as f64;
        corrections[frame] =
            [0, 1].map(|axis| end_gap[axis] * blend + end_velocity_gap[axis] * velocity_blend);
    }
    corrections
}

/// Soften a looping track, so a nudge from a neighbor reads as a fish easing aside and
/// not as a twitch.
fn smooth_track(points: &mut Vec<[f64; 2]>) {
    const PASSES: usize = 4;
    let total = points.len();
    for _ in 0..PASSES {
        *points = (0..total)
            .map(|frame| {
                let before = points[(frame + total - 1) % total];
                let here = points[frame];
                let after = points[(frame + 1) % total];
                [0, 1].map(|axis| (before[axis] + 2.0 * here[axis] + after[axis]) / 4.0)
            })
            .collect();
    }
}

/// Close a list of positions recorded for frames zero through the loop's length, then
/// drop the repeated last frame.
fn close_loop_points(points: &mut Vec<[f64; 2]>) {
    let frames_total = points.len() - 1;
    let end_gap = [0, 1].map(|axis| points[0][axis] - points[frames_total][axis]);
    let velocity_gap = [0, 1].map(|axis| {
        (points[1][axis] - points[0][axis])
            - (points[frames_total][axis] - points[frames_total - 1][axis])
    });
    for (point, correction) in
        points
            .iter_mut()
            .zip(closing_correction(frames_total, end_gap, velocity_gap))
    {
        point[0] += correction[0];
        point[1] += correction[1];
    }
    points.pop();
}

/// Close a fry's track, then work out how it faces and how its tail swings from where
/// it ended up, so its body always points the way it is moving.
fn close_fry(record: &mut Vec<FryPlace>) {
    const TURN_LOOKBACK_FRAMES: usize = 7;
    // Fry arrive home all but exactly, so only a position nudge is needed, and matching
    // velocity as well would swing them much farther than the gap it closes.
    let mut heads: Vec<[f64; 2]> = record.iter().map(|place| place.head).collect();
    let frames_total = heads.len() - 1;
    let end_gap = [0, 1].map(|axis| heads[0][axis] - heads[frames_total][axis]);
    for (head, correction) in
        heads
            .iter_mut()
            .zip(closing_correction(frames_total, end_gap, [0.0; 2]))
    {
        head[0] += correction[0];
        head[1] += correction[1];
    }
    heads.pop();
    smooth_track(&mut heads);
    let frame_total = heads.len();
    let total_tail_phase = record[frame_total].wag;
    let whole_beats = (total_tail_phase / TAU).round().max(1.0);
    let phase_scale = whole_beats * TAU / total_tail_phase.max(1e-9);
    // The simulation's own heading: it is what collisions were judged against.
    let mut forwards: Vec<[f64; 2]> = record[..frame_total]
        .iter()
        .map(|place| place.forward)
        .collect();
    // Soften what is left, around the loop, so a push never flips a fish in one frame.
    for _ in 0..3 {
        forwards = (0..frame_total)
            .map(|frame| {
                let before = forwards[(frame + frame_total - 1) % frame_total];
                let after = forwards[(frame + 1) % frame_total];
                let here = forwards[frame];
                normalize([
                    before[0] + 2.0 * here[0] + after[0],
                    before[1] + 2.0 * here[1] + after[1],
                ])
            })
            .collect();
    }
    let headings: Vec<f64> = forwards.iter().map(|[x, y]| y.atan2(*x)).collect();
    *record = (0..frame_total)
        .map(|frame| FryPlace {
            head: heads[frame],
            forward: forwards[frame],
            turn_radians: angle_between(
                headings[(frame + frame_total - TURN_LOOKBACK_FRAMES) % frame_total],
                headings[frame],
            ),
            wag: (record[frame].wag * phase_scale).sin(),
        })
        .collect();
}
