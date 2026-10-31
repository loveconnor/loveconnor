//! The day's movement, chosen from recent activity by one ordered set of rules.

use crate::koi::{KoiDevelopment, RecentActivity};

/// One movement per run.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Behavior {
    /// Sits low in the water, blue-tinted, barely moving.
    DeepRest,
    /// Hovers in place a little deeper, with a slow beat and glide.
    Rest,
    /// Hovers at the surface and gulps.
    SurfaceFeeding,
    /// Two fast laps of strong bursts and glides.
    Dart,
    /// One slow lap; the everyday swim.
    Cruise,
}

const DEEP_REST_QUIET_DAYS: u32 = 60;
const REST_QUIET_DAYS: u32 = 14;
/// Totals whose crossing earns a week of darting.
const DART_MILESTONES: [u64; 6] = [1_000, 5_000, 10_000, 25_000, 50_000, 100_000];
/// A busy week is compared with at most this many of the latest complete weeks.
const DART_COMPARED_WEEKS: usize = 52;
/// A top tenth means little with fewer complete weeks than this.
const DART_MIN_COMPLETE_WEEKS: usize = 10;
/// A busy week is matched or beaten by at most one in this many compared weeks.
const DART_WEEK_SHARE_DENOMINATOR: usize = 10;
const DAYS_PER_WEEK: usize = 7;

/// Choose the movement for the day after the observation date. A nightly run judges
/// the day that just ended, and the koi shows the result all of the next day. The
/// first matching rule wins:
///
/// | Behavior | When the observation date |
/// | --- | --- |
/// | Deep rest | ends DEEP_REST_QUIET_DAYS or more quiet days |
/// | Rest | ends REST_QUIET_DAYS or more quiet days |
/// | Surface feeding | is the first day with contributions in its Monday-Sunday week |
/// | Dart | falls within seven days of the total crossing a milestone |
/// | Dart | is a Sunday that finishes a busy week |
/// | Cruise | is anything else |
///
/// A finished week is busy when at most one in ten of the latest
/// DART_COMPARED_WEEKS complete weeks before it match or beat its total, and at
/// least DART_MIN_COMPLETE_WEEKS weeks are compared. So a steady account cruises,
/// and the koi darts at most one day a week beyond milestones. This reads no clock
/// and changes no growth, colors, or marks.
pub fn select_behavior(koi_development: &KoiDevelopment) -> Behavior {
    let recent_activity = &koi_development.recent_activity;
    if recent_activity.quiet_day_count >= DEEP_REST_QUIET_DAYS {
        Behavior::DeepRest
    } else if recent_activity.quiet_day_count >= REST_QUIET_DAYS {
        Behavior::Rest
    } else if is_first_contribution_of_week(recent_activity) {
        Behavior::SurfaceFeeding
    } else if has_crossed_milestone(koi_development.total_contributions, recent_activity)
        || is_finished_busy_week(recent_activity)
    {
        Behavior::Dart
    } else {
        Behavior::Cruise
    }
}

fn is_first_contribution_of_week(recent_activity: &RecentActivity) -> bool {
    matches!(
        recent_activity.observation_week_contributions.split_last(),
        Some((&observed_day, earlier_days))
            if observed_day > 0 && earlier_days.iter().all(|&day_contributions| day_contributions == 0)
    )
}

fn has_crossed_milestone(total_contributions: u64, recent_activity: &RecentActivity) -> bool {
    let earlier_total = total_contributions.saturating_sub(recent_activity.recent_contributions);
    DART_MILESTONES
        .iter()
        .any(|&milestone| earlier_total < milestone && milestone <= total_contributions)
}

fn is_finished_busy_week(recent_activity: &RecentActivity) -> bool {
    let finished_week = &recent_activity.observation_week_contributions;
    let complete_weeks = &recent_activity.complete_week_contributions;
    let compared_weeks =
        &complete_weeks[complete_weeks.len().saturating_sub(DART_COMPARED_WEEKS)..];
    let week_total: u64 = finished_week.iter().sum();
    let matching_week_count = compared_weeks
        .iter()
        .filter(|&&week_contributions| week_contributions >= week_total)
        .count();
    finished_week.len() == DAYS_PER_WEEK
        && compared_weeks.len() >= DART_MIN_COMPLETE_WEEKS
        && matching_week_count * DART_WEEK_SHARE_DENOMINATOR <= compared_weeks.len()
}
