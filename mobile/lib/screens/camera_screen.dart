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

  /// What the capture button says while busy, e.g. "Analyzing… (2/3)".
  String _busyLabel = '';

  /// False if setting a flash mode threw, i.e. the camera has no flash.
  /// (On Android the plugin doesn't throw for that, so this only catches
  /// platforms that report it, such as iOS.)
  bool _hasFlash = false;
  bool _torchOn = false;

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

  /// True while _setupCamera() runs. The first initialize() can show the
  /// permission dialog, which sends the app through inactive -> resumed;
  /// without this, the resume would start a second setup alongside it.
  bool _settingUp = false;

  Future<void> _setupCamera() async {
    if (_settingUp) {
      return;
    }
    _settingUp = true;
    try {
      final cameras = await availableCameras();
      if (cameras.isEmpty) {
        if (mounted) {
          setState(() {
            _error = 'No camera found on this device.';
            _isInitializing = false;
          });
        }
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

      // The plugin defaults to FlashMode.auto, which fires in low light.
      // Flash causes glare and colour shifts on skin that hurt predictions,
      // so it stays off unless the user turns on the steady torch.
      var hasFlash = true;
      try {
        await controller.setFlashMode(FlashMode.off);
      } on CameraException catch (e) {
        debugPrint('No usable flash: ${e.description}');
        hasFlash = false;
      }

      if (!mounted) {
        // Left the screen while the camera was starting.
        await controller.dispose();
        return;
      }
      setState(() {
        _controller = controller;
        _hasFlash = hasFlash;
        _torchOn = false;
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
    } finally {
      _settingUp = false;
    }
  }

  @override
  void didChangeAppLifecycleState(AppLifecycleState state) {
    final controller = _controller;
    if (state == AppLifecycleState.inactive) {
      if (controller == null || !controller.value.isInitialized) {
        return;
      }
      // Going to the background: the torch goes off with the camera.
      setState(() {
        _controller = null;
        _torchOn = false;
      });
      _release(controller);
    } else if (state == AppLifecycleState.resumed && controller == null) {
      _setupCamera();
    }
  }

  /// Turns the torch off (if it might be on), then disposes [controller].
  /// Doesn't touch widget state, so it is safe to call from dispose().
  Future<void> _release(CameraController controller) async {
    try {
      if (controller.value.isInitialized &&
          controller.value.flashMode != FlashMode.off) {
        await controller.setFlashMode(FlashMode.off);
      }
    } on CameraException catch (e) {
      debugPrint('Could not turn the torch off: ${e.description}');
    }
    await controller.dispose();
  }

  Future<void> _toggleTorch() async {
    final controller = _controller;
    if (controller == null || !controller.value.isInitialized) {
      return;
    }
    final next = _torchOn ? FlashMode.off : FlashMode.torch;
    try {
      await controller.setFlashMode(next);
      if (mounted) {
        setState(() => _torchOn = next == FlashMode.torch);
      }
    } on CameraException catch (e) {
      debugPrint('Torch failed: ${e.description}');
      if (!mounted) {
        return;
      }
      setState(() {
        _hasFlash = false;
        _torchOn = false;
      });
      ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(content: Text('This camera has no torch.')),
      );
    }
  }

  Future<void> _turnTorchOff() async {
    final controller = _controller;
    if (!_torchOn || controller == null) {
      return;
    }
    try {
      await controller.setFlashMode(FlashMode.off);
    } on CameraException catch (e) {
      debugPrint('Could not turn the torch off: ${e.description}');
    }
    if (mounted) {
      setState(() => _torchOn = false);
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

    setState(() {
      _isCapturing = true;
      _busyLabel = 'Capturing…';
    });

    try {
      final XFile shot = await controller.takePicture();
      // Not needed for analysis, and the result screen covers this one.
      await _turnTorchOff();

      final docsDir = await getApplicationDocumentsDirectory();
      final scansDir = Directory(p.join(docsDir.path, 'scans'));
      if (!await scansDir.exists()) {
        await scansDir.create(recursive: true);
      }

      final filename = 'scan_${DateTime.now().millisecondsSinceEpoch}.jpg';
      final saved = await File(shot.path).copy(p.join(scansDir.path, filename));

      if (mounted) {
        setState(() => _busyLabel = 'Analyzing…');
      }
      final result = await _inference.classify(
        saved,
        onProgress: (current, total) {
          if (mounted) {
            setState(() => _busyLabel = 'Analyzing… ($current/$total)');
          }
        },
      );

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
    final controller = _controller;
    _controller = null;
    if (controller != null) {
      // Torch off first, then dispose (fire-and-forget: dispose() is sync).
      _release(controller);
    }
    // The ensemble is app-wide: main.dart disposes it, not this screen.
    super.dispose();
  }

  bool get _cameraReady => _controller?.value.isInitialized ?? false;

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: Colors.black,
      appBar: AppBar(
        title: const Text('Capture scan '),
        actions: [
          if (_cameraReady && _hasFlash)
            IconButton(
              onPressed: _isCapturing ? null : _toggleTorch,
              tooltip: _torchOn ? 'Turn torch off' : 'Turn torch on',
              isSelected: _torchOn,
              icon: const Icon(Icons.flashlight_off),
              selectedIcon: const Icon(Icons.flashlight_on),
            ),
        ],
      ),
      body: _buildBody(),
      floatingActionButton: _cameraReady ? _buildCaptureButton() : null,
      floatingActionButtonLocation: FloatingActionButtonLocation.centerFloat,
    );
  }

  Widget _buildCaptureButton() {
    if (_modelsError != null) {
      return const SizedBox.shrink();
    }
    if (!_modelsReady || _isCapturing) {
      // Disabled until the ensemble is ready, and while a scan is analyzed.
      return FloatingActionButton.extended(
        onPressed: null,
        icon: const SizedBox.square(
          dimension: 20,
          child: CircularProgressIndicator(strokeWidth: 2),
        ),
        label: Text(_isCapturing ? _busyLabel : 'Loading models…'),
      );
    }
    return FloatingActionButton.large(
      onPressed: _capture,
      tooltip: 'Take scan',
      child: const Icon(Icons.camera_alt),
    );
  }

  Widget _buildBody() {
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
    final controller = _controller;
    // Also covers the moment between going to the background and resuming.
    if (_isInitializing || controller == null) {
      return const Center(child: CircularProgressIndicator());
    }

    return Stack(
      alignment: Alignment.center,
      children: [
        Center(child: CameraPreview(controller)),
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
