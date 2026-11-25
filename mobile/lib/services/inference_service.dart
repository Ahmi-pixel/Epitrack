import 'dart:io';
import 'dart:math';
import '../models/prediction.dart';

/// Contract for skin-condition classification.
/// Implemented by [StubInferenceService] now, by a TFLite-backed
/// service once the trained model is available.
abstract class InferenceService {
  Future<void> load();
  Future<InferenceResult> classify(File image);
  void dispose();
}

class StubInferenceService implements InferenceService {
  static const List<String> labels = [
    'Acne', 'Eczema', 'Psoriasis', 'Melanoma', 'Benign Nevus',
    'Tinea', 'Vitiligo', 'Rosacea', 'Seborrheic Keratosis',
    'Basal Cell Carcinoma',
  ];

  final _random = Random();
  bool _loaded = false;

  @override
  Future<void> load() async {
    await Future.delayed(const Duration(milliseconds: 300));
    _loaded = true;
  }

  @override
  Future<InferenceResult> classify(File image) async {
    if (!_loaded) throw StateError('Call load() before classify().');

    final stopwatch = Stopwatch()..start();
    await Future.delayed(const Duration(milliseconds: 600));

    // Fake but plausible: a few random scores, normalized, sorted.
    final scores = List.generate(labels.length, (_) => _random.nextDouble());
    final total = scores.reduce((a, b) => a + b);

    final predictions = <Prediction>[
      for (var i = 0; i < labels.length; i++)
        Prediction(label: labels[i], confidence: scores[i] / total),
    ]..sort((a, b) => b.confidence.compareTo(a.confidence));

    stopwatch.stop();
    return InferenceResult(
      predictions: predictions.take(3).toList(),
      inferenceTimeMs: stopwatch.elapsedMilliseconds,
    );
  }

  @override
  void dispose() {
    _loaded = false;
  }
}