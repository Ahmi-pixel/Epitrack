import 'dart:io';

import 'package:camera/camera.dart';
import 'package:flutter/material.dart';
import 'package:path_provider/path_provider.dart';
import 'package:path/path.dart' as p;
import 'package:provider/provider.dart';

import '../services/inference_service.dart';
import 'result_screen.dart';

class CameraScreen extends StatefulWidget {
  const CameraScreen({super.key});

  @override
  State<CameraScreen> createState() => _CameraScreenState();
}

class _CameraScreenState extends State<CameraScreen>
    with WidgetsBindingObserver {
  CameraController? _controller;
  bool _isInitializing = true;
  bool _isCapturing = false;
  String? _error;

  /// The app-wide ensemble from main.dart. It started loading at app
  /// startup; this screen only waits for it.
  late final InferenceService _inference;
  bool _modelsReady = false;
  String? _modelsError;

  @override
  void initState() {
    super.initState();
    WidgetsBinding.instance.addObserver(this);
    _setupCamera();
    _inference = context.read<InferenceService>();
    _waitForModels();
  }

  Future<void> _waitForModels() async {
    try {
      await _inference.load();
      if (mounted) {
        setState(() => _modelsReady = true);
      }
    } catch (e) {
      debugPrint('Model loading failed: $e');
      if (mounted) {
        setState(() => _modelsError = 'Could not load the models: $e');
      }
    }
  }

  Future<void> _setupCamera() async {
    try {
      final cameras = await availableCameras();
      if (cameras.isEmpty) {
        setState(() {
          _error = 'No camera found on this device.';
          _isInitializing = false;
        });
        return;
      }

      final back = cameras.firstWhere(
        (c) => c.lensDirection == CameraLensDirection.back,
        orElse: () => cameras.first,
      );

      final controller = CameraController(
        back,
        ResolutionPreset.high,
        enableAudio: false,
      );
      await controller.initialize();

      if (!mounted) {
        return;
      }
      setState(() {
        _controller = controller;
        _isInitializing = false;
      });
    } on CameraException catch (e) {
      if (!mounted) {
        return;
      }
      setState(() {
        _error = 'Camera error: ${e.description}';
        _isInitializing = false;
      });
    }
  }

  @override
  void didChangeAppLifecycleState(AppLifecycleState state) {
    final controller = _controller;
    if (controller == null || !controller.value.isInitialized) {
      return;
    }

    if (state == AppLifecycleState.inactive) {
      controller.dispose();
      _controller = null;
    } else if (state == AppLifecycleState.resumed) {
      _setupCamera();
    }
  }

  Future<void> _capture() async {
    final controller = _controller;
    if (controller == null ||
        !controller.value.isInitialized ||
        _isCapturing ||
        !_modelsReady) {
      return;
    }

    setState(() => _isCapturing = true);

    try {
      final XFile shot = await controller.takePicture();

      final docsDir = await getApplicationDocumentsDirectory();
      final scansDir = Directory(p.join(docsDir.path, 'scans'));
      if (!await scansDir.exists()) {
        await scansDir.create(recursive: true);
      }

      final filename = 'scan_${DateTime.now().millisecondsSinceEpoch}.jpg';
      final saved = await File(shot.path).copy(p.join(scansDir.path, filename));

      final result = await _inference.classify(saved);

      if (!mounted) {
        return;
      }
      Navigator.of(context).push(
        MaterialPageRoute<void>(
          builder: (_) => ResultScreen(image: saved, result: result),
        ),
      );
    } on CameraException catch (e) {
      if (!mounted) {
        return;
      }
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(content: Text('Capture failed: ${e.description}')),
      );
    } catch (e) {
      // E.g. an undecodable image or a model error.
      debugPrint('Analysis failed: $e');
      if (!mounted) {
        return;
      }
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(content: Text('Analysis failed: $e')),
      );
    } finally {
      if (mounted) {
        setState(() => _isCapturing = false);
      }
    }
  }

  @override
  void dispose() {
    WidgetsBinding.instance.removeObserver(this);
    _controller?.dispose();
    // The ensemble is app-wide: main.dart disposes it, not this screen.
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: Colors.black,
      appBar: AppBar(title: const Text('Capture scan ')),
      body: _buildBody(),
      floatingActionButton: (_controller?.value.isInitialized ?? false)
          ? _buildCaptureButton()
          : null,
      floatingActionButtonLocation: FloatingActionButtonLocation.centerFloat,
    );
  }

  Widget _buildCaptureButton() {
    if (_modelsError != null) {
      return const SizedBox.shrink();
    }
    if (!_modelsReady) {
      // Disabled until the ensemble is loaded and warmed up.
      return FloatingActionButton.extended(
        onPressed: null,
        icon: const SizedBox.square(
          dimension: 20,
          child: CircularProgressIndicator(strokeWidth: 2),
        ),
        label: const Text('Loading models…'),
      );
    }
    return FloatingActionButton.large(
      onPressed: _isCapturing ? null : _capture,
      child: _isCapturing
          ? const CircularProgressIndicator()
          : const Icon(Icons.camera_alt),
    );
  }

  Widget _buildBody() {
    if (_isInitializing) {
      return const Center(child: CircularProgressIndicator());
    }
    final error = _error ?? _modelsError;
    if (error != null) {
      return Center(
        child: Padding(
          padding: const EdgeInsets.all(24),
          child: Text(
            error,
            style: const TextStyle(color: Colors.white),
            textAlign: TextAlign.center,
          ),
        ),
      );
    }

    return Stack(
      alignment: Alignment.center,
      children: [
        Center(child: CameraPreview(_controller!)),
        // Framing guide — keeps capture distance consistent between visits.
        IgnorePointer(
          child: Container(
            width: 260,
            height: 260,
            decoration: BoxDecoration(
              border: Border.all(color: Colors.white70, width: 2),
              borderRadius: BorderRadius.circular(12),
            ),
          ),
        ),
      ],
    );
  }
}
