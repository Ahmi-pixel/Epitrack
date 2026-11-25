/// Pure ensemble logic: config/label parsing, probability averaging,
/// malignancy and category scoring. No Flutter or TFLite imports, so it can
/// be unit-tested directly.
library;

import '../models/prediction.dart';

/// The parsed contents of assets/models/epitrack_config.json.
class EnsembleConfig {
  /// Model file names, relative to assets/models/.
  final List<String> modelFiles;
  final double malignancyThreshold;
  final List<String> malignantClasses;

  /// Category name -> the class labels it groups. Insertion order is kept,
  /// which decides ties in [topCategory].
  final Map<String, List<String>> categoryGroups;
  final String modelVersion;

  const EnsembleConfig({
    required this.modelFiles,
    required this.malignancyThreshold,
    required this.malignantClasses,
    required this.categoryGroups,
    required this.modelVersion,
  });

  /// Throws [FormatException] naming the first missing or malformed field.
  factory EnsembleConfig.fromJson(Map<String, dynamic> json) {
    List<String> stringList(Object? value, String field) {
      if (value is! List || value.any((e) => e is! String)) {
        throw FormatException(
          'epitrack_config.json: "$field" must be a list of strings',
        );
      }
      return List<String>.unmodifiable(value);
    }

    final models = stringList(json['models'], 'models');
    if (models.isEmpty) {
      throw const FormatException('epitrack_config.json: "models" is empty');
    }

    final threshold = json['malignancy_threshold'];
    if (threshold is! num) {
      throw const FormatException(
        'epitrack_config.json: "malignancy_threshold" must be a number',
      );
    }

    final groups = json['category_groups'];
    if (groups is! Map<String, dynamic> || groups.isEmpty) {
      throw const FormatException(
        'epitrack_config.json: "category_groups" must be a non-empty object',
      );
    }

    final version = json['model_version'];
    if (version is! String) {
      throw const FormatException(
        'epitrack_config.json: "model_version" must be a string',
      );
    }

    return EnsembleConfig(
      modelFiles: models,
      malignancyThreshold: threshold.toDouble(),
      malignantClasses: stringList(
        json['malignant_classes'],
        'malignant_classes',
      ),
      categoryGroups: {
        for (final entry in groups.entries)
          entry.key: stringList(entry.value, 'category_groups.${entry.key}'),
      },
      modelVersion: version,
    );
  }

  /// Throws [FormatException] if a class named here isn't one of [labels],
  /// or a class is in more than one category (it would be counted twice).
  void checkAgainstLabels(List<String> labels) {
    final known = labels.toSet();
    for (final name in malignantClasses) {
      if (!known.contains(name)) {
        throw FormatException(
          'epitrack_config.json: malignant class "$name" is not in '
          'class_indices_15.json',
        );
      }
    }
    final seen = <String, String>{};
    for (final MapEntry(key: category, value: names)
        in categoryGroups.entries) {
      for (final name in names) {
        if (!known.contains(name)) {
          throw FormatException(
            'epitrack_config.json: class "$name" in category "$category" '
            'is not in class_indices_15.json',
          );
        }
        final previous = seen[name];
        if (previous != null) {
          throw FormatException(
            'epitrack_config.json: class "$name" is in both "$previous" '
            'and "$category"',
          );
        }
        seen[name] = category;
      }
    }
  }
}

/// Turns class_indices_15.json (label -> index) into labels in output
/// order. Throws [FormatException] unless there are exactly
/// [expectedCount] labels with indices 0..expectedCount-1, each used once.
List<String> labelsFromClassIndices(
  Map<String, dynamic> json, {
  int expectedCount = 15,
}) {
  if (json.length != expectedCount) {
    throw FormatException(
      'class_indices_15.json: expected $expectedCount classes, '
      'found ${json.length}',
    );
  }
  final labels = List<String?>.filled(expectedCount, null);
  for (final MapEntry(key: label, value: index) in json.entries) {
    if (index is! int || index < 0 || index >= expectedCount) {
      throw FormatException(
        'class_indices_15.json: "$label" has index $index, expected an '
        'integer 0-${expectedCount - 1}',
      );
    }
    if (labels[index] != null) {
      throw FormatException(
        'class_indices_15.json: index $index is used by both '
        '"${labels[index]}" and "$label"',
      );
    }
    labels[index] = label;
  }
  // Every slot is filled: expectedCount distinct indices in range.
  return List<String>.unmodifiable(labels.cast<String>());
}

/// Equal-weight mean of each model's probabilities, class by class.
List<double> averageProbabilities(List<List<double>> perModel) {
  if (perModel.isEmpty) {
    throw ArgumentError('No model outputs to average');
  }
  final n = perModel.first.length;
  if (perModel.any((p) => p.length != n)) {
    throw ArgumentError('Model outputs have different lengths');
  }
  return List<double>.generate(n, (k) {
    var sum = 0.0;
    for (final p in perModel) {
      sum += p[k];
    }
    return sum / perModel.length;
  }, growable: false);
}

/// The result of scoring one averaged probability vector.
class EnsembleOutcome {
  /// All classes, highest probability first.
  final List<Prediction> predictions;
  final double malignancyScore;
  final bool malignant;

  /// Category name -> summed probability, in config order.
  final Map<String, double> categoryScores;
  final String category;

  const EnsembleOutcome({
    required this.predictions,
    required this.malignancyScore,
    required this.malignant,
    required this.categoryScores,
    required this.category,
  });
}

/// Scores [probabilities] (in [labels] order) against [config].
EnsembleOutcome scoreEnsemble(
  List<double> probabilities,
  List<String> labels,
  EnsembleConfig config,
) {
  if (probabilities.length != labels.length) {
    throw ArgumentError(
      '${probabilities.length} probabilities for ${labels.length} labels',
    );
  }
  final byLabel = {
    for (var k = 0; k < labels.length; k++) labels[k]: probabilities[k],
  };

  final malignancy = config.malignantClasses.fold<double>(
    0,
    (sum, name) => sum + byLabel[name]!,
  );

  final categoryScores = {
    for (final MapEntry(key: category, value: names)
        in config.categoryGroups.entries)
      category: names.fold<double>(0, (sum, name) => sum + byLabel[name]!),
  };

  return EnsembleOutcome(
    predictions: [
      for (var k = 0; k < labels.length; k++)
        Prediction(label: labels[k], confidence: probabilities[k]),
    ]..sort((a, b) => b.confidence.compareTo(a.confidence)),
    malignancyScore: malignancy,
    malignant: malignancy >= config.malignancyThreshold,
    categoryScores: categoryScores,
    category: topCategory(categoryScores),
  );
}

/// The category with the highest score. On a tie the first one wins.
String topCategory(Map<String, double> scores) {
  if (scores.isEmpty) {
    throw ArgumentError('No categories');
  }
  var best = scores.entries.first;
  for (final entry in scores.entries.skip(1)) {
    if (entry.value > best.value) {
      best = entry;
    }
  }
  return best.key;
}
