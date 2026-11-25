import 'dart:io';

import 'package:flutter/material.dart';
import 'package:provider/provider.dart';

import '../models/prediction.dart';
import '../models/scan.dart';
import '../services/database_service.dart';
import '../widgets/lesion_picker_sheet.dart';

class ResultScreen extends StatefulWidget {
  final File image;
  final InferenceResult result;

  const ResultScreen({super.key, required this.image, required this.result});

  @override
  State<ResultScreen> createState() => _ResultScreenState();
}

class _ResultScreenState extends State<ResultScreen> {
  bool _isSaving = false;

  Future<void> _saveToLesion() async {
    final db = context.read<DatabaseService>();
    final lesion = await showLesionPickerSheet(context);
    if (lesion == null || !mounted) {
      return;
    }

    setState(() => _isSaving = true);

    try {
      await db.insertScan(
        Scan(
          lesionId: lesion.id!,
          imagePath: widget.image.path,
          capturedAt: DateTime.now(),
          predictions: widget.result.predictions,
          inferenceTimeMs: widget.result.inferenceTimeMs,
          category: widget.result.category,
          malignant: widget.result.malignant,
          malignancyScore: widget.result.malignancyScore,
          modelVersion: widget.result.modelVersion,
        ),
      );

      if (!mounted) {
        return;
      }
      ScaffoldMessenger.of(context)
          .showSnackBar(SnackBar(content: Text('Saved to ${lesion.name}')));
      Navigator.of(context).pop();
    } finally {
      if (mounted) {
        setState(() => _isSaving = false);
      }
    }
  }

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    final result = widget.result;
    final category = result.category;

    return Scaffold(
      appBar: AppBar(title: const Text('Scan result')),
      body: SafeArea(
        child: ListView(
          padding: const EdgeInsets.all(16),
          children: [
            ClipRRect(
              borderRadius: BorderRadius.circular(12),
              child: Image.file(
                widget.image,
                height: 280,
                width: double.infinity,
                fit: BoxFit.cover,
              ),
            ),
            const SizedBox(height: 24),
            if (category != null) ...[
              Text('Category', style: theme.textTheme.labelLarge),
              Text(category, style: theme.textTheme.headlineMedium),
              const SizedBox(height: 16),
            ],
            if (result.malignant ?? false) ...[
              const _MalignancyWarning(),
              const SizedBox(height: 16),
            ],
            Text('Top match', style: theme.textTheme.labelLarge),
            Text(
              result.top.label,
              style: category == null
                  ? theme.textTheme.headlineMedium
                  : theme.textTheme.titleLarge,
            ),
            const SizedBox(height: 16),
            for (final prediction in result.top3)
              _PredictionRow(prediction: prediction),
            const SizedBox(height: 16),
            Card(
              child: Padding(
                padding: const EdgeInsets.all(16),
                child: Row(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Icon(Icons.info_outline, color: theme.colorScheme.primary),
                    const SizedBox(width: 12),
                    const Expanded(
                      child: Text(
                        'This is a screening aid, not a diagnosis. '
                        'Please see a dermatologist about any skin concern.',
                      ),
                    ),
                  ],
                ),
              ),
            ),
            const SizedBox(height: 8),
            Text(
              'Inference time: ${result.inferenceTimeMs} ms',
              style: theme.textTheme.bodySmall,
            ),
            const SizedBox(height: 24),
            FilledButton(
              onPressed: _isSaving ? null : _saveToLesion,
              child: const Text('Save to lesion'),
            ),
            const SizedBox(height: 8),
            FilledButton(
              onPressed: () => Navigator.of(context).pop(),
              child: const Text('Scan again'),
            ),
          ],
        ),
      ),
    );
  }
}

class _MalignancyWarning extends StatelessWidget {
  const _MalignancyWarning();

  @override
  Widget build(BuildContext context) {
    final colors = Theme.of(context).colorScheme;
    return Card(
      color: colors.errorContainer,
      child: Padding(
        padding: const EdgeInsets.all(16),
        child: Row(
          children: [
            Icon(Icons.warning_amber_rounded, color: colors.onErrorContainer),
            const SizedBox(width: 12),
            Expanded(
              child: Text(
                'Possible malignant features. Please see a dermatologist.',
                style: TextStyle(
                  color: colors.onErrorContainer,
                  fontWeight: FontWeight.w600,
                ),
              ),
            ),
          ],
        ),
      ),
    );
  }
}

class _PredictionRow extends StatelessWidget {
  final Prediction prediction;

  const _PredictionRow({required this.prediction});

  @override
  Widget build(BuildContext context) {
    return Padding(
      padding: const EdgeInsets.symmetric(vertical: 6),
      child: Column(
        children: [
          Row(
            children: [
              Expanded(child: Text(prediction.label)),
              Text('${(prediction.confidence * 100).toStringAsFixed(1)}%'),
            ],
          ),
          const SizedBox(height: 4),
          LinearProgressIndicator(value: prediction.confidence),
        ],
      ),
    );
  }
}
