import 'dart:convert';
import 'dart:io';
import 'dart:typed_data';

import 'package:flutter/foundation.dart';
import 'package:flutter/services.dart' show rootBundle;
import 'package:image/image.dart' as img;
import 'package:tflite_flutter/tflite_flutter.dart';

import '../models/prediction.dart';
import 'ensemble_math.dart';
import 'inference_service.dart';

const _modelDir = 'assets/models/';
const _configAsset = '${_modelDir}epitrack_config.json';
const _labelsAsset = '${_modelDir}class_indices_15.json';
const _inputSize = 380;
const _classCount = 15;

/// Decodes the image at [path] and turns it into the models' input:
/// 380x380 RGB, NHWC, raw 0-255 float32 values.
///
/// Top-level so compute() can run it in a background isolate.
Float32List preprocessImageFile(String path) {
  final decoded = img.decodeImage(File(path).readAsBytesSync());
  if (decoded == null) {
    throw FormatException('Not a decodable image: $path');
  }
  // copyResize applies the JPEG's EXIF orientation while resizing, so the
  // result is upright whichever way the phone was held.
  var resized = img.copyResize(
    decoded,
    width: _inputSize,
    height: _inputSize,
    interpolation: img.Interpolation.linear,
  );
  // r/g/b below need 8-bit RGB: on a grayscale image g and b read as 0.
  // Camera JPEGs already are, so this normally does nothing.
  if (resized.format != img.Format.uint8 ||
      resized.numChannels != 3 ||
      resized.hasPalette) {
    resized = resized.convert(format: img.Format.uint8, numChannels: 3);
  }

  // NHWC with raw 0-255 values. The models have their preprocessing built
  // in, so no dividing by 255 and no mean/std normalization here.
  final input = Float32List(_inputSize * _inputSize * 3);
  var i = 0;
  for (var y = 0; y < _inputSize; y++) {
    for (var x = 0; x < _inputSize; x++) {
      final pixel = resized.getPixel(x, y);
      input[i++] = pixel.r.toDouble();
      input[i++] = pixel.g.toDouble();
      input[i++] = pixel.b.toDouble();
    }
  }
  return input;
}

/// One loaded model of the ensemble.
class _EnsembleMember {
  final String file;
  final Interpreter interpreter;

  /// Runs [interpreter] in a background isolate. Null on the GPU, whose
  /// delegate has to be used from the thread that created it.
  final IsolateInterpreter? isolate;

  /// Must outlive [interpreter]: closing it doesn't free the delegate.
  final GpuDelegateV2? gpuDelegate;
  final String backend;

  _EnsembleMember({
    required this.file,
    required this.interpreter,
    required this.isolate,
    required this.gpuDelegate,
    required this.backend,
  });

  /// [input] is the flat NHWC bytes of the [1, 380, 380, 3] tensor, [output]
  /// 15 float32s. Passing flat buffers instead of nested lists matters:
  /// tflite_flutter converts nested lists one element at a time, which is
  /// very slow for 433,200 floats.
  Future<void> run(ByteBuffer input, ByteBuffer output) async {
    final isolate = this.isolate;
    if (isolate != null) {
      await isolate.run(input, output);
    } else {
      interpreter.run(input, output);
    }
  }

  Future<void> close() async {
    await isolate?.close();
    interpreter.close();
    gpuDelegate?.delete();
  }
}

/// Classifies scans on-device with the ensemble listed in
/// epitrack_config.json: every model gets the same preprocessed input and
/// their 15 output probabilities are averaged equally.
///
/// Create one for the whole app: load() reads and warms up every model,
/// which takes several seconds and a few hundred MB.
///
/// Experiments: the defaults come from build-time defines, e.g.
///   flutter run --dart-define=EPITRACK_THREADS=4
///   flutter run --dart-define=EPITRACK_GPU=true
///   flutter run --dart-define=EPITRACK_GPU=true --dart-define=EPITRACK_GPU_FP16=true
class TFLiteInferenceService implements InferenceService {
  /// Try the GPU delegate first, per model. Falls back to CPU for a model
  /// whose GPU setup fails.
  final bool useGpu;

  /// Let the GPU delegate compute in 16-bit floats (isPrecisionLossAllowed).
  /// Only applies when [useGpu] is true.
  final bool gpuFp16;

  /// CPU threads per model. The models run one after another, so this is
  /// also the most threads in use at once.
  final int threads;

  TFLiteInferenceService({
    this.useGpu = const bool.fromEnvironment('EPITRACK_GPU'),
    this.gpuFp16 = const bool.fromEnvironment('EPITRACK_GPU_FP16'),
    this.threads = const int.fromEnvironment(
      'EPITRACK_THREADS',
      defaultValue: 2,
    ),
  });

  Future<void>? _loading;
  bool _disposed = false;
  final List<_EnsembleMember> _members = [];
  late EnsembleConfig _config;
  late List<String> _labels;

  /// Labels in model output order, from class_indices_15.json.
  List<String> get labels => _labels;

  String get _requestedBackend =>
      useGpu ? (gpuFp16 ? 'gpu-fp16' : 'gpu') : 'cpu';

  /// Loads and warms up every model. Safe to call more than once: later
  /// calls return the same future, so callers can all await readiness.
  @override
  Future<void> load() => _loading ??= _load();

  Future<void> _load() async {
    final stopwatch = Stopwatch()..start();
    _config = EnsembleConfig.fromJson(
      jsonDecode(await rootBundle.loadString(_configAsset))
          as Map<String, dynamic>,
    );
    _labels = labelsFromClassIndices(
      jsonDecode(await rootBundle.loadString(_labelsAsset))
          as Map<String, dynamic>,
      expectedCount: _classCount,
    );
    _config.checkAgainstLabels(_labels);

    try {
      // One at a time, to keep peak memory and CPU use down.
      for (final file in _config.modelFiles) {
        final member = await _loadMember(file);
        _members.add(member);
        if (_disposed) {
          throw StateError('Disposed while loading');
        }
      }
    } catch (_) {
      await _closeMembers();
      rethrow;
    }

    debugPrint(
      'TFLite ensemble ready: ${_members.length} models '
      '(${_config.modelVersion}), threads=$threads, '
      'requested $_requestedBackend, '
      'in ${stopwatch.elapsedMilliseconds} ms',
    );
  }

  Future<_EnsembleMember> _loadMember(String file) async {
    final Uint8List bytes;
    try {
      final data = await rootBundle.load('$_modelDir$file');
      bytes = data.buffer.asUint8List(data.offsetInBytes, data.lengthInBytes);
    } catch (e) {
      throw StateError(
        'Model "$file" is listed in epitrack_config.json but '
        '$_modelDir$file is missing from the app bundle. Copy it into '
        'mobile/$_modelDir and rebuild. ($e)',
      );
    }

    if (useGpu) {
      final gpu = await _tryGpuMember(file, bytes);
      if (gpu != null) {
        return gpu;
      }
    }

    final interpreter = Interpreter.fromBuffer(
      bytes,
      options: InterpreterOptions()..threads = threads,
    );
    IsolateInterpreter? isolate;
    try {
      _checkTensors(file, interpreter);
      isolate = await IsolateInterpreter.create(
        address: interpreter.address,
        debugName: 'tflite:$file',
      );
      final member = _EnsembleMember(
        file: file,
        interpreter: interpreter,
        isolate: isolate,
        gpuDelegate: null,
        backend: 'cpu',
      );
      await _warmUp(member);
      return member;
    } catch (_) {
      await isolate?.close();
      interpreter.close();
      rethrow;
    }
  }

  /// Null if the GPU delegate can't be used for this model. Its problems
  /// usually surface when the interpreter is created or on its first run,
  /// not in GpuDelegateV2(), so the warm-up is inside the try too.
  Future<_EnsembleMember?> _tryGpuMember(String file, Uint8List bytes) async {
    GpuDelegateV2? delegate;
    Interpreter? interpreter;
    try {
      // With no options the delegate uses its defaults (fp32 compute).
      if (gpuFp16) {
        final options = GpuDelegateOptionsV2(isPrecisionLossAllowed: true);
        // The delegate copies its options, so they can be freed right away.
        delegate = GpuDelegateV2(options: options);
        options.delete();
      } else {
        delegate = GpuDelegateV2();
      }
      interpreter = Interpreter.fromBuffer(
        bytes,
        options: InterpreterOptions()
          ..threads = threads
          ..addDelegate(delegate),
      );
      _checkTensors(file, interpreter);
      final member = _EnsembleMember(
        file: file,
        interpreter: interpreter,
        isolate: null,
        gpuDelegate: delegate,
        backend: _requestedBackend,
      );
      await _warmUp(member);
      return member;
    } catch (e) {
      debugPrint('TFLite: $file: GPU delegate failed, using CPU: $e');
      interpreter?.close();
      delegate?.delete();
      return null;
    }
  }

  /// Throws if [interpreter]'s tensors aren't the ones classify() is
  /// written for. The caller closes [interpreter] on failure.
  void _checkTensors(String file, Interpreter interpreter) {
    final input = interpreter.getInputTensor(0);
    final output = interpreter.getOutputTensor(0);
    debugPrint(
      'TFLite $file: input ${input.shape} ${input.type}, '
      'output ${output.shape} ${output.type}',
    );
    if (!listEquals(input.shape, [1, _inputSize, _inputSize, 3]) ||
        !listEquals(output.shape, [1, _classCount])) {
      throw StateError(
        'Model "$file" has input ${input.shape} and output ${output.shape}; '
        'expected [1, $_inputSize, $_inputSize, 3] and [1, $_classCount]',
      );
    }
  }

  /// One dummy run on an all-zeros input, so the first real scan's timing
  /// doesn't include one-off setup (tensor allocation, GPU shader compile).
  Future<void> _warmUp(_EnsembleMember member) async {
    final stopwatch = Stopwatch()..start();
    await member.run(
      Float32List(_inputSize * _inputSize * 3).buffer,
      Float32List(_classCount).buffer,
    );
    debugPrint(
      'TFLite ${member.file}: backend=${member.backend}, '
      'warm-up ${stopwatch.elapsedMilliseconds} ms',
    );
  }

  @override
  Future<InferenceResult> classify(File image) async {
    await load();
    if (_disposed) {
      throw StateError('classify() after dispose()');
    }

    final total = Stopwatch()..start();
    // Decoding a 12 MP JPEG takes a while: keep it off the UI thread.
    final input = await compute(preprocessImageFile, image.path);
    final preprocessMs = total.elapsedMilliseconds;
    debugPrint('[TIMING] preprocessing=$preprocessMs');

    // Sequentially, not in parallel: each model already uses [threads]
    // threads, and running them together would oversubscribe the CPU.
    final perModel = <List<double>>[];
    for (final member in _members) {
      final stopwatch = Stopwatch()..start();
      final output = Float32List(_classCount);
      await member.run(input.buffer, output.buffer);
      perModel.add(output.toList(growable: false));
      debugPrint('[TIMING] ${member.file}=${stopwatch.elapsedMilliseconds}');
    }

    // The models' outputs are already softmaxed.
    final outcome = scoreEnsemble(
      averageProbabilities(perModel),
      _labels,
      _config,
    );
    total.stop();
    debugPrint('[TIMING] total=${total.elapsedMilliseconds}');
    debugPrint(
      '[RESULT] category=${outcome.category} '
      'malignancy=${outcome.malignancyScore.toStringAsFixed(4)} '
      '(threshold ${_config.malignancyThreshold}, '
      'flagged=${outcome.malignant})',
    );
    for (final p in outcome.predictions.take(3)) {
      debugPrint(
        '[RESULT]   ${p.label}: ${(p.confidence * 100).toStringAsFixed(2)}%',
      );
    }

    return InferenceResult(
      predictions: outcome.predictions,
      inferenceTimeMs: total.elapsedMilliseconds,
      category: outcome.category,
      malignant: outcome.malignant,
      malignancyScore: outcome.malignancyScore,
      modelVersion: _config.modelVersion,
    );
  }

  Future<void> _closeMembers() async {
    final members = List.of(_members);
    _members.clear();
    for (final member in members) {
      await member.close();
    }
  }

  /// Frees every interpreter. On Android the OS usually kills the process
  /// without this running, which frees the memory anyway.
  @override
  void dispose() {
    if (_disposed) {
      return;
    }
    _disposed = true;
    // A load still in progress cleans up when it sees _disposed.
    _closeMembers();
  }
}
