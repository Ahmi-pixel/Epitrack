class Prediction {
  final String label;
  final double confidence;

  const Prediction({required this.label, required this.confidence});

  @override
  String toString() => '$label (${(confidence * 100).toStringAsFixed(1)}%)';
}

class InferenceResult {
  /// Highest confidence first. The ensemble returns all 15 classes.
  final List<Prediction> predictions;
  final int inferenceTimeMs;

  /// The ensemble's extras. Null from a service that doesn't compute them.
  final String? category;
  final bool? malignant;
  final double? malignancyScore;
  final String? modelVersion;

  const InferenceResult({
    required this.predictions,
    required this.inferenceTimeMs,
    this.category,
    this.malignant,
    this.malignancyScore,
    this.modelVersion,
  });

  Prediction get top => predictions.first;

  /// The (up to) three most likely classes.
  List<Prediction> get top3 => predictions.take(3).toList();
}
