import 'dart:convert';
import 'dart:io';

import 'package:epitrack/services/ensemble_math.dart';
import 'package:flutter_test/flutter_test.dart';

Map<String, dynamic> _readJson(String path) =>
    jsonDecode(File(path).readAsStringSync()) as Map<String, dynamic>;

void main() {
  group('real asset files', () {
    // flutter test runs from the package root (mobile/).
    final config = EnsembleConfig.fromJson(
      _readJson('assets/models/epitrack_config.json'),
    );
    final labels = labelsFromClassIndices(
      _readJson('assets/models/class_indices_15.json'),
    );

    test('config lists the three models and its version', () {
      expect(config.modelFiles, [
        'b4_15_fp16.tflite',
        'densenet_15_fp16.tflite',
        'convnext_15_fp16.tflite',
      ]);
      expect(config.malignancyThreshold, 0.26);
      expect(config.modelVersion, '15class-v1');
    });

    test('every listed model file exists', () {
      for (final file in config.modelFiles) {
        expect(File('assets/models/$file').existsSync(), isTrue,
            reason: '$file missing from assets/models/');
      }
    });

    test('labels are the 15 classes in index order', () {
      expect(labels, hasLength(15));
      expect(labels.first, 'Acne and Rosacea Photos');
      expect(labels[12], 'melanoma');
      expect(labels.last, 'vascular_lesion');
    });

    test('config class names all match the labels', () {
      expect(() => config.checkAgainstLabels(labels), returnsNormally);
    });
  });

  group('labelsFromClassIndices', () {
    test('orders by index', () {
      expect(
        labelsFromClassIndices({'b': 1, 'a': 0, 'c': 2}, expectedCount: 3),
        ['a', 'b', 'c'],
      );
    });

    test('rejects the wrong count', () {
      expect(
        () => labelsFromClassIndices({'a': 0, 'b': 1}, expectedCount: 3),
        throwsFormatException,
      );
    });

    test('rejects a duplicate or out-of-range index', () {
      expect(
        () => labelsFromClassIndices({'a': 0, 'b': 0, 'c': 2},
            expectedCount: 3),
        throwsFormatException,
      );
      expect(
        () => labelsFromClassIndices({'a': 0, 'b': 1, 'c': 3},
            expectedCount: 3),
        throwsFormatException,
      );
    });
  });

  group('EnsembleConfig', () {
    Map<String, dynamic> json() => {
      'models': ['m.tflite'],
      'malignancy_threshold': 0.5,
      'malignant_classes': ['bad'],
      'category_groups': {
        'Malignant': ['bad'],
        'Other': ['ok', 'meh'],
      },
      'model_version': 'v',
    };

    test('rejects a missing field', () {
      expect(
        () => EnsembleConfig.fromJson(json()..remove('models')),
        throwsFormatException,
      );
    });

    test('rejects unknown class names', () {
      final config = EnsembleConfig.fromJson(json());
      expect(
        () => config.checkAgainstLabels(['bad', 'ok']),
        throwsFormatException,
      );
    });

    test('rejects a class in two categories', () {
      final config = EnsembleConfig.fromJson(
        json()
          ..['category_groups'] = {
            'A': ['bad', 'ok'],
            'B': ['ok', 'meh'],
          },
      );
      expect(
        () => config.checkAgainstLabels(['bad', 'ok', 'meh']),
        throwsFormatException,
      );
    });
  });

  group('averageProbabilities', () {
    test('is the equal-weight mean per class', () {
      final avg = averageProbabilities([
        [0.6, 0.3, 0.1],
        [0.2, 0.5, 0.3],
        [0.1, 0.1, 0.8],
      ]);
      expect(avg[0], closeTo(0.3, 1e-12));
      expect(avg[1], closeTo(0.3, 1e-12));
      expect(avg[2], closeTo(0.4, 1e-12));
    });

    test('rejects mismatched lengths', () {
      expect(
        () => averageProbabilities([
          [0.5, 0.5],
          [1.0],
        ]),
        throwsArgumentError,
      );
    });
  });

  group('scoreEnsemble', () {
    const labels = ['bad1', 'bad2', 'ok', 'meh'];
    EnsembleConfig config(double threshold) => EnsembleConfig(
      modelFiles: const ['m.tflite'],
      malignancyThreshold: threshold,
      malignantClasses: const ['bad1', 'bad2'],
      categoryGroups: const {
        'Malignant': ['bad1', 'bad2'],
        'Benign': ['ok'],
        'Rash': ['meh'],
      },
      modelVersion: 'v',
    );

    test('sums malignant classes and flags at the threshold', () {
      // bad1 + bad2 = 0.26 exactly (in binary, as close as doubles allow).
      final outcome = scoreEnsemble([0.16, 0.10, 0.5, 0.24], labels,
          config(0.16 + 0.10));
      expect(outcome.malignancyScore, closeTo(0.26, 1e-12));
      expect(outcome.malignant, isTrue);
    });

    test('does not flag below the threshold', () {
      final outcome = scoreEnsemble([0.1, 0.1, 0.5, 0.3], labels, config(0.26));
      expect(outcome.malignancyScore, closeTo(0.2, 1e-12));
      expect(outcome.malignant, isFalse);
    });

    test('picks the category with the highest sum, not the top class', () {
      // 'ok' is the single most likely class (0.4), but the two malignant
      // classes together (0.5) outweigh it.
      final outcome = scoreEnsemble([0.25, 0.25, 0.4, 0.1], labels,
          config(0.26));
      expect(outcome.predictions.first.label, 'ok');
      expect(outcome.category, 'Malignant');
      expect(outcome.categoryScores['Malignant'], closeTo(0.5, 1e-12));
    });

    test('returns every class, highest first', () {
      final outcome = scoreEnsemble([0.1, 0.2, 0.3, 0.4], labels, config(0.9));
      expect(outcome.predictions.map((p) => p.label),
          ['meh', 'ok', 'bad2', 'bad1']);
    });
  });

  test('topCategory keeps the first on a tie', () {
    expect(topCategory({'A': 0.5, 'B': 0.5}), 'A');
  });
}
