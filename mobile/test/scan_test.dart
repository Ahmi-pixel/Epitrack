import 'dart:convert';

import 'package:epitrack/models/prediction.dart';
import 'package:epitrack/models/scan.dart';
import 'package:flutter_test/flutter_test.dart';

Map<String, Object?> _row(String predictionsJson) => {
  'id': 1,
  'lesion_id': 2,
  'image_path': '/scans/a.jpg',
  'captured_at': 1700000000000,
  'predictions': predictionsJson,
  'inference_time_ms': 1234,
  'note': null,
};

void main() {
  test('reads an old scan whose predictions are a top-3 list', () {
    final scan = Scan.fromMap(_row(jsonEncode([
      {'label': 'nevus', 'confidence': 0.7},
      {'label': 'melanoma', 'confidence': 0.2},
      {'label': 'dermatofibroma', 'confidence': 0.05},
    ])));

    expect(scan.predictions, hasLength(3));
    expect(scan.top.label, 'nevus');
    expect(scan.top.confidence, 0.7);
    expect(scan.category, isNull);
    expect(scan.malignant, isNull);
    expect(scan.malignancyScore, isNull);
    expect(scan.modelVersion, isNull);
  });

  test('round-trips an ensemble scan through toMap/fromMap', () {
    final predictions = [
      for (var i = 0; i < 15; i++)
        Prediction(label: 'class$i', confidence: (15 - i) / 120),
    ];
    final original = Scan(
      id: 1,
      lesionId: 2,
      imagePath: '/scans/a.jpg',
      capturedAt: DateTime.fromMillisecondsSinceEpoch(1700000000000),
      predictions: predictions,
      inferenceTimeMs: 1234,
      category: 'Benign growth',
      malignant: true,
      malignancyScore: 0.31,
      modelVersion: '15class-v1',
    );

    final map = original.toMap();
    final stored = jsonDecode(map['predictions'] as String) as Map;
    expect(stored['format'], 2);
    expect(stored['predictions'], hasLength(15));

    final scan = Scan.fromMap(map);
    expect(scan.predictions, hasLength(15));
    expect(scan.top.label, 'class0');
    expect(scan.category, 'Benign growth');
    expect(scan.malignant, isTrue);
    expect(scan.malignancyScore, 0.31);
    expect(scan.modelVersion, '15class-v1');
  });
}
