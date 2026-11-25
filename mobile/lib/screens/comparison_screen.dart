import 'package:flutter/material.dart';

import '../models/lesion.dart';
import '../models/prediction.dart';
import '../models/scan.dart';
import '../utils/dates.dart';
import '../widgets/scan_image.dart';

class ComparisonScreen extends StatelessWidget {
  final Lesion lesion;
  final Scan earlier;
  final Scan later;

  const ComparisonScreen({
    super.key,
    required this.lesion,
    required this.earlier,
    required this.later,
  });

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    final days = calendarDaysBetween(earlier.capturedAt, later.capturedAt);

    return Scaffold(
      appBar: AppBar(
        title: Column(
          mainAxisSize: MainAxisSize.min,
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            const Text('Compare scans'),
            Text(lesion.name, style: theme.textTheme.bodySmall),
          ],
        ),
      ),
      body: SafeArea(
        child: ListView(
          padding: const EdgeInsets.all(16),
          children: [
            Row(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Expanded(child: _ScanColumn(scan: earlier)),
                const SizedBox(width: 12),
                Expanded(child: _ScanColumn(scan: later)),
              ],
            ),
            const SizedBox(height: 24),
            Text(switch (days) {
              0 => 'Taken on the same day',
              1 => '1 day apart',
              _ => '$days days apart',
            }, style: theme.textTheme.titleMedium),
            const SizedBox(height: 16),
            Card(
              child: Padding(
                padding: const EdgeInsets.all(16),
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Text('Change', style: theme.textTheme.labelLarge),
                    const SizedBox(height: 4),
                    Text(
                      _changeSummary(earlier.top, later.top),
                      style: theme.textTheme.bodyLarge,
                    ),
                    const SizedBox(height: 8),
                    Text(
                      'Confidence is how sure the model is of its label, not a '
                      'measure of severity.',
                      style: theme.textTheme.bodySmall,
                    ),
                  ],
                ),
              ),
            ),
          ],
        ),
      ),
    );
  }
}

class _ScanColumn extends StatelessWidget {
  final Scan scan;

  const _ScanColumn({required this.scan});

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    final top = scan.top;

    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        AspectRatio(
          aspectRatio: 1,
          child: ClipRRect(
            borderRadius: BorderRadius.circular(12),
            child: ScanImage(path: scan.imagePath),
          ),
        ),
        const SizedBox(height: 8),
        Text(
          scanDateFormat.format(scan.capturedAt),
          style: theme.textTheme.labelLarge,
        ),
        const SizedBox(height: 4),
        Text(top.label, style: theme.textTheme.titleMedium),
        Text('${_percent(top.confidence)}%'),
      ],
    );
  }
}

/// Only compares confidences when both scans' top predictions name the same
/// condition: confidences for different labels aren't comparable.
String _changeSummary(Prediction earlier, Prediction later) {
  if (earlier.label != later.label) {
    return 'The top prediction changed from ${earlier.label} to '
        "${later.label}, so their confidences can't be compared directly.";
  }

  // Use the rounded figures shown on screen so the delta always matches them.
  final before = _percent(earlier.confidence);
  final after = _percent(later.confidence);
  final delta = after - before;
  final direction = delta > 0 ? 'up' : 'down';
  final change = switch (delta.abs()) {
    0 => 'no change',
    1 => '$direction 1 point',
    final points => '$direction $points points',
  };
  return '${earlier.label}: $before% → $after%, $change';
}

int _percent(double confidence) => (confidence * 100).round();
