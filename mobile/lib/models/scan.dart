import 'dart:convert';

import 'prediction.dart';

class Scan {
  final int? id;
  final int lesionId;
  final String imagePath;
  final DateTime capturedAt;

  /// Highest confidence first. All 15 classes for ensemble scans, only the
  /// top 3 for scans saved before the ensemble.
  final List<Prediction> predictions;
  final int inferenceTimeMs;
  final String? note;

  /// Null for scans saved before the ensemble.
  final String? category;
  final bool? malignant;
  final double? malignancyScore;
  final String? modelVersion;

  const Scan({
    this.id,
    required this.lesionId,
    required this.imagePath,
    required this.capturedAt,
    required this.predictions,
    required this.inferenceTimeMs,
    this.note,
    this.category,
    this.malignant,
    this.malignancyScore,
    this.modelVersion,
  });

  /// Reads both formats of the `predictions` column:
  /// - old: a JSON list of {label, confidence} (top 3);
  /// - format 2: an object with "predictions" (all classes) plus
  ///   "category", "malignant", "malignancy_score" and "model_version".
  factory Scan.fromMap(Map<String, Object?> map) {
    final decoded = jsonDecode(map['predictions'] as String);
    final Map<String, dynamic>? extras = decoded is Map<String, dynamic>
        ? decoded
        : null;
    final items = (extras?['predictions'] ?? decoded) as List;

    return Scan(
      id: map['id'] as int?,
      lesionId: map['lesion_id'] as int,
      imagePath: map['image_path'] as String,
      capturedAt: DateTime.fromMillisecondsSinceEpoch(
        map['captured_at'] as int,
      ),
      predictions: [
        for (final item in items)
          Prediction(
            label: item['label'] as String,
            confidence: (item['confidence'] as num).toDouble(),
          ),
      ],
      inferenceTimeMs: map['inference_time_ms'] as int,
      note: map['note'] as String?,
      category: extras?['category'] as String?,
      malignant: extras?['malignant'] as bool?,
      malignancyScore: (extras?['malignancy_score'] as num?)?.toDouble(),
      modelVersion: extras?['model_version'] as String?,
    );
  }

  Prediction get top => predictions.first;

  /// Always writes format 2. The table schema is unchanged: everything new
  /// lives inside the `predictions` TEXT column.
  Map<String, Object?> toMap() => {
    if (id != null) 'id': id,
    'lesion_id': lesionId,
    'image_path': imagePath,
    'captured_at': capturedAt.millisecondsSinceEpoch,
    'predictions': jsonEncode({
      'format': 2,
      'model_version': modelVersion,
      'category': category,
      'malignant': malignant,
      'malignancy_score': malignancyScore,
      'predictions': [
        for (final prediction in predictions)
          {'label': prediction.label, 'confidence': prediction.confidence},
      ],
    }),
    'inference_time_ms': inferenceTimeMs,
    'note': note,
  };
}
