import 'package:flutter/material.dart';
import 'package:provider/provider.dart';

import '../models/lesion.dart';
import '../models/scan.dart';
import '../services/database_service.dart';
import '../services/image_files.dart';
import '../utils/dates.dart';
import '../widgets/confirm_delete_dialog.dart';
import '../widgets/scan_image.dart';
import 'camera_screen.dart';
import 'comparison_screen.dart';

class LesionDetailScreen extends StatefulWidget {
  final Lesion lesion;

  const LesionDetailScreen({super.key, required this.lesion});

  @override
  State<LesionDetailScreen> createState() => _LesionDetailScreenState();
}

class _LesionDetailScreenState extends State<LesionDetailScreen> {
  List<Scan>? _scans;
  bool _isSelecting = false;
  final Set<int> _selectedIds = {};

  @override
  void initState() {
    super.initState();
    _load();
  }

  Future<void> _load() async {
    final scans = await context.read<DatabaseService>().getScansForLesion(
      widget.lesion.id!,
    );
    if (!mounted) {
      return;
    }
    setState(() => _scans = scans);
  }

  Future<void> _openCamera() async {
    await Navigator.of(context)
        .push(MaterialPageRoute<void>(builder: (_) => const CameraScreen()));
    if (!mounted) {
      return;
    }
    // A scan taken from here may have been saved to this lesion.
    await _load();
  }

  Future<void> _deleteScan(Scan scan) async {
    final db = context.read<DatabaseService>();
    final confirmed = await confirmDelete(
      context,
      title: 'Delete scan?',
      message:
          'The scan from ${scanDateFormat.format(scan.capturedAt)} and its '
          "photo will be permanently deleted. This can't be undone.",
    );
    if (!confirmed) {
      return;
    }

    await db.deleteScan(scan.id!);
    await deleteImageFile(scan.imagePath);
    if (!mounted) {
      return;
    }
    await _load();
  }

  void _startSelecting() {
    setState(() => _isSelecting = true);
  }

  void _stopSelecting() {
    setState(() {
      _isSelecting = false;
      _selectedIds.clear();
    });
  }

  void _toggleSelected(Scan scan) {
    final id = scan.id!;
    if (_selectedIds.contains(id)) {
      setState(() => _selectedIds.remove(id));
    } else if (_selectedIds.length < 2) {
      setState(() => _selectedIds.add(id));
    } else {
      ScaffoldMessenger.of(context)
        ..hideCurrentSnackBar()
        ..showSnackBar(
          const SnackBar(
            content: Text('You can compare two scans. Deselect one first.'),
          ),
        );
    }
  }

  void _compare() {
    // _scans is oldest first, so filtering it keeps the pair in that order.
    final pair = _scans!.where((s) => _selectedIds.contains(s.id)).toList();
    Navigator.of(context).push(
      MaterialPageRoute<void>(
        builder: (_) => ComparisonScreen(
          lesion: widget.lesion,
          earlier: pair[0],
          later: pair[1],
        ),
      ),
    );
  }

  @override
  Widget build(BuildContext context) {
    final location = widget.lesion.bodyLocation;
    final scans = _scans;

    return PopScope(
      // While selecting, Back leaves selection mode rather than the screen.
      canPop: !_isSelecting,
      onPopInvokedWithResult: (didPop, _) {
        if (!didPop) {
          _stopSelecting();
        }
      },
      child: Scaffold(
        appBar: AppBar(
          title: Column(
            mainAxisSize: MainAxisSize.min,
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Text(widget.lesion.name),
              if (location != null)
                Text(location, style: Theme.of(context).textTheme.bodySmall),
            ],
          ),
          actions: [
            if (!_isSelecting && scans != null && scans.length >= 2)
              IconButton(
                onPressed: _startSelecting,
                tooltip: 'Compare scans',
                icon: const Icon(Icons.compare),
              ),
          ],
        ),
        body: _buildBody(),
        floatingActionButton: _isSelecting
            ? null
            : FloatingActionButton(
                onPressed: _openCamera,
                tooltip: 'New scan',
                child: const Icon(Icons.camera_alt),
              ),
        bottomNavigationBar: _isSelecting ? _buildSelectionBar() : null,
      ),
    );
  }

  Widget _buildBody() {
    final scans = _scans;
    if (scans == null) {
      return const Center(child: CircularProgressIndicator());
    }
    if (scans.isEmpty) {
      return const Center(
        child: Padding(
          padding: EdgeInsets.all(24),
          child: Text(
            'No scans for this lesion yet.\n'
            'Tap the camera button to take one, then save it to this lesion.',
            textAlign: TextAlign.center,
          ),
        ),
      );
    }

    return ListView.builder(
      // Keeps the last card clear of the floating action button.
      padding: const EdgeInsets.fromLTRB(16, 8, 16, 88),
      itemCount: scans.length,
      itemBuilder: (context, index) {
        final scan = scans[index];
        return _ScanCard(
          scan: scan,
          previous: index == 0 ? null : scans[index - 1],
          isSelecting: _isSelecting,
          isSelected: _selectedIds.contains(scan.id),
          onTap: () => _toggleSelected(scan),
          onDelete: () => _deleteScan(scan),
        );
      },
    );
  }

  Widget _buildSelectionBar() {
    final count = _selectedIds.length;

    return BottomAppBar(
      child: Row(
        children: [
          Expanded(
            child: Text(
              count == 0 ? 'Select two scans' : '$count of 2 selected',
            ),
          ),
          TextButton(onPressed: _stopSelecting, child: const Text('Cancel')),
          const SizedBox(width: 8),
          FilledButton(
            onPressed: count == 2 ? _compare : null,
            child: const Text('Compare'),
          ),
        ],
      ),
    );
  }
}

class _ScanCard extends StatelessWidget {
  final Scan scan;
  final Scan? previous;
  final bool isSelecting;
  final bool isSelected;
  final VoidCallback onTap;
  final VoidCallback onDelete;

  const _ScanCard({
    required this.scan,
    this.previous,
    required this.isSelecting,
    required this.isSelected,
    required this.onTap,
    required this.onDelete,
  });

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    final colors = theme.colorScheme;
    final previous = this.previous;
    final top = scan.top;

    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        if (previous != null)
          Padding(
            padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 4),
            child: Text(
              _gapLabel(
                calendarDaysBetween(previous.capturedAt, scan.capturedAt),
              ),
              style: theme.textTheme.bodySmall,
            ),
          ),
        Card(
          clipBehavior: Clip.antiAlias,
          color: isSelected ? colors.secondaryContainer : null,
          shape: isSelected
              ? RoundedRectangleBorder(
                  borderRadius: BorderRadius.circular(12),
                  side: BorderSide(color: colors.primary, width: 2),
                )
              : null,
          child: InkWell(
            onTap: isSelecting ? onTap : null,
            child: Padding(
              padding: const EdgeInsets.all(12),
              child: Row(
                children: [
                  ClipRRect(
                    borderRadius: BorderRadius.circular(8),
                    child: ScanImage(
                      path: scan.imagePath,
                      width: 80,
                      height: 80,
                    ),
                  ),
                  const SizedBox(width: 16),
                  Expanded(
                    child: Column(
                      crossAxisAlignment: CrossAxisAlignment.start,
                      children: [
                        Text(
                          scanDateFormat.format(scan.capturedAt),
                          style: theme.textTheme.labelLarge,
                        ),
                        const SizedBox(height: 4),
                        Text(top.label, style: theme.textTheme.titleMedium),
                        Text('${(top.confidence * 100).toStringAsFixed(1)}%'),
                      ],
                    ),
                  ),
                  // Same 48px footprint either way, so the row doesn't shift.
                  if (isSelecting)
                    Padding(
                      padding: const EdgeInsets.all(12),
                      child: Icon(
                        isSelected
                            ? Icons.check_circle
                            : Icons.radio_button_unchecked,
                        color: isSelected
                            ? colors.primary
                            : colors.onSurfaceVariant,
                      ),
                    )
                  else
                    IconButton(
                      onPressed: onDelete,
                      tooltip: 'Delete scan',
                      icon: const Icon(Icons.delete_outline),
                    ),
                ],
              ),
            ),
          ),
        ),
      ],
    );
  }
}

String _gapLabel(int days) => switch (days) {
  0 => 'Same day',
  1 => '1 day later',
  _ => '$days days later',
};
