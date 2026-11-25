import 'package:flutter/material.dart';
import 'package:provider/provider.dart';

import '../models/lesion.dart';
import '../services/database_service.dart';
import '../services/image_files.dart';
import '../widgets/confirm_delete_dialog.dart';
import 'camera_screen.dart';
import 'lesion_detail_screen.dart';

class LesionListScreen extends StatefulWidget {
  const LesionListScreen({super.key});

  @override
  State<LesionListScreen> createState() => _LesionListScreenState();
}

class _LesionListScreenState extends State<LesionListScreen> {
  List<Lesion>? _lesions;
  Map<int, int> _scanCounts = {};

  @override
  void initState() {
    super.initState();
    _load();
  }

  Future<void> _load() async {
    final db = context.read<DatabaseService>();
    final lesions = await db.getLesions();
    // DatabaseService has no count query, so count each lesion's scans.
    final counts = await Future.wait([
      for (final lesion in lesions)
        db.getScansForLesion(lesion.id!).then((scans) => scans.length),
    ]);

    if (!mounted) {
      return;
    }
    setState(() {
      _lesions = lesions;
      _scanCounts = {
        for (var i = 0; i < lesions.length; i++) lesions[i].id!: counts[i],
      };
    });
  }

  /// Pushes [screen], then reloads so new lesions and scan counts show up.
  Future<void> _openScreen(Widget screen) async {
    await Navigator.of(context)
        .push(MaterialPageRoute<void>(builder: (_) => screen));
    if (!mounted) {
      return;
    }
    await _load();
  }

  Future<void> _deleteLesion(Lesion lesion) async {
    final db = context.read<DatabaseService>();
    final scans = switch (_scanCounts[lesion.id] ?? 0) {
      0 => 'It has no scans.',
      1 => 'Its scan and photo will be deleted too.',
      final n => 'All $n of its scans and their photos will be deleted too.',
    };
    final confirmed = await confirmDelete(
      context,
      title: 'Delete "${lesion.name}"?',
      message: "$scans This can't be undone.",
    );
    if (!confirmed) {
      return;
    }

    // Read the image paths before the cascade removes the scan rows. Files go
    // last, so a failed database delete never leaves rows whose photos are
    // already gone.
    final lesionScans = await db.getScansForLesion(lesion.id!);
    await db.deleteLesion(lesion.id!);
    await Future.wait([
      for (final scan in lesionScans) deleteImageFile(scan.imagePath),
    ]);
    if (!mounted) {
      return;
    }
    await _load();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text('EpiTrack')),
      body: _buildBody(),
      floatingActionButton: FloatingActionButton(
        onPressed: () => _openScreen(const CameraScreen()),
        tooltip: 'New scan',
        child: const Icon(Icons.camera_alt),
      ),
    );
  }

  Widget _buildBody() {
    final lesions = _lesions;
    if (lesions == null) {
      return const Center(child: CircularProgressIndicator());
    }
    if (lesions.isEmpty) {
      return const Center(
        child: Padding(
          padding: EdgeInsets.all(24),
          child: Text(
            'No lesions tracked yet.\n'
            'Tap the camera button to take a scan, then save it to a new '
            'lesion to start tracking it over time.',
            textAlign: TextAlign.center,
          ),
        ),
      );
    }

    return ListView.builder(
      // Keeps the last tile clear of the floating action button.
      padding: const EdgeInsets.only(bottom: 88),
      itemCount: lesions.length,
      itemBuilder: (context, index) {
        final lesion = lesions[index];
        final location = lesion.bodyLocation;
        final count = _scanCounts[lesion.id] ?? 0;
        return ListTile(
          title: Text(lesion.name),
          subtitle: location == null ? null : Text(location),
          trailing: Row(
            mainAxisSize: MainAxisSize.min,
            children: [
              Text(count == 1 ? '1 scan' : '$count scans'),
              PopupMenuButton<void>(
                tooltip: 'More options',
                itemBuilder: (_) => [
                  PopupMenuItem<void>(
                    onTap: () => _deleteLesion(lesion),
                    child: const Text('Delete'),
                  ),
                ],
              ),
            ],
          ),
          onTap: () => _openScreen(LesionDetailScreen(lesion: lesion)),
        );
      },
    );
  }
}
