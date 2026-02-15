use std::collections::VecDeque;

use egui::{Color32, Pos2, Rect, Stroke, Ui, Vec2};

use crate::state::MetricsSnapshot;

/// Draws a simple time-series line chart from metrics history.
pub fn metric_line_chart(
    ui: &mut Ui,
    _id: &str,
    label: &str,
    history: &VecDeque<MetricsSnapshot>,
    value_fn: fn(&MetricsSnapshot) -> f64,
    color: Color32,
    size: Vec2,
) {
    if history.len() < 2 {
        ui.allocate_space(size);
        ui.put(
            Rect::from_min_size(
                ui.min_rect().left_bottom() - Vec2::new(0.0, size.y),
                size,
            ),
            egui::Label::new(format!("{}: waiting for data...", label)),
        );
        return;
    }

    let (response, painter) =
        ui.allocate_painter(size, egui::Sense::hover());
    let rect = response.rect;

    // Background
    painter.rect_filled(rect, 4.0, Color32::from_gray(30));

    // Collect values
    let values: Vec<f64> = history.iter().map(|s| value_fn(s)).collect();
    let max_val = values
        .iter()
        .cloned()
        .fold(f64::NEG_INFINITY, f64::max)
        .max(1.0);

    let margin = 4.0;
    let chart_rect = Rect::from_min_max(
        Pos2::new(rect.min.x + margin, rect.min.y + margin + 14.0),
        Pos2::new(rect.max.x - margin, rect.max.y - margin),
    );

    // Draw label and current value
    let current = values.last().copied().unwrap_or(0.0);
    painter.text(
        Pos2::new(rect.min.x + margin, rect.min.y + margin),
        egui::Align2::LEFT_TOP,
        format!("{}: {:.1}", label, current),
        egui::FontId::proportional(11.0),
        Color32::WHITE,
    );

    // Draw max value
    painter.text(
        Pos2::new(rect.max.x - margin, rect.min.y + margin),
        egui::Align2::RIGHT_TOP,
        format!("max: {:.1}", max_val),
        egui::FontId::proportional(9.0),
        Color32::from_gray(120),
    );

    if chart_rect.width() <= 0.0 || chart_rect.height() <= 0.0 {
        return;
    }

    // Draw grid lines
    for i in 1..4 {
        let y = chart_rect.min.y + chart_rect.height() * (i as f32 / 4.0);
        painter.line_segment(
            [
                Pos2::new(chart_rect.min.x, y),
                Pos2::new(chart_rect.max.x, y),
            ],
            Stroke::new(0.5, Color32::from_gray(50)),
        );
    }

    // Draw line
    let n = values.len();
    if n >= 2 {
        let points: Vec<Pos2> = values
            .iter()
            .enumerate()
            .map(|(i, &v)| {
                let x = chart_rect.min.x
                    + chart_rect.width() * (i as f32 / (n - 1) as f32);
                let y = chart_rect.max.y
                    - chart_rect.height() * (v as f32 / max_val as f32);
                Pos2::new(x, y)
            })
            .collect();

        for window in points.windows(2) {
            painter.line_segment(
                [window[0], window[1]],
                Stroke::new(1.5, color),
            );
        }
    }
}

/// Compute a rate value from two consecutive snapshots.
pub fn rate_from_snapshots(
    prev: &MetricsSnapshot,
    curr: &MetricsSnapshot,
    value_fn: fn(&MetricsSnapshot) -> u64,
) -> f64 {
    let dt = curr
        .timestamp
        .duration_since(prev.timestamp)
        .as_secs_f64();
    if dt > 0.0 {
        (value_fn(curr).saturating_sub(value_fn(prev))) as f64 / dt
    } else {
        0.0
    }
}
