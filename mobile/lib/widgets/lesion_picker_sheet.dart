import 'package:flutter/material.dart';
import 'package:provider/provider.dart';

import '../models/lesion.dart';
import '../services/database_service.dart';

/// Shows a sheet for picking an existing lesion or creating a new one.
/// Completes with the chosen [Lesion], or null if the sheet is dismissed.
Future<Lesion?> showLesionPickerSheet(BuildContext context) {
  return showModalBottomSheet<Lesion>(
    context: context,
    builder: (_) => const LesionPickerSheet(),
  );
}

class LesionPickerSheet extends StatefulWidget {
  const LesionPickerSheet({super.key});

  @override
  State<LesionPickerSheet> createState() => _LesionPickerSheetState();
}

class _LesionPickerSheetState extends State<LesionPickerSheet> {
  List<Lesion>? _lesions;

  @override
  void initState() {
    super.initState();
    _load();
  }

  Future<void> _load() async {
    final lesions = await context.read<DatabaseService>().getLesions();
    if (!mounted) {
      return;
    }
    setState(() => _lesions = lesions);
  }

  Future<void> _createLesion() async {
    final db = context.read<DatabaseService>();
    final draft = await showDialog<Lesion>(
      context: context,
      builder: (_) => const _NewLesionDialog(),
    );
    if (draft == null) {
      return;
    }

    final id = await db.insertLesion(draft);
    if (!mounted) {
      return;
    }
    Navigator.of(context).pop(
      Lesion(
        id: id,
        name: draft.name,
        bodyLocation: draft.bodyLocation,
        createdAt: draft.createdAt,
      ),
    );
  }

  @override
  Widget build(BuildContext context) {
    final lesions = _lesions;

    return SafeArea(
      child: ListView(
        shrinkWrap: true,
        children: [
          Padding(
            padding: const EdgeInsets.fromLTRB(16, 16, 16, 8),
            child: Text(
              'Choose a lesion',
              style: Theme.of(context).textTheme.titleMedium,
            ),
          ),
          ListTile(
            leading: const Icon(Icons.add),
            title: const Text('New lesion'),
            onTap: _createLesion,
          ),
          if (lesions == null)
            const Padding(
              padding: EdgeInsets.all(16),
              child: Center(child: CircularProgressIndicator()),
            )
          else
            for (final lesion in lesions)
              ListTile(
                title: Text(lesion.name),
                subtitle: lesion.bodyLocation == null
                    ? null
                    : Text(lesion.bodyLocation!),
                onTap: () => Navigator.of(context).pop(lesion),
              ),
        ],
      ),
    );
  }
}

/// Asks for a name and optional body location. Pops an unsaved [Lesion].
class _NewLesionDialog extends StatefulWidget {
  const _NewLesionDialog();

  @override
  State<_NewLesionDialog> createState() => _NewLesionDialogState();
}

class _NewLesionDialogState extends State<_NewLesionDialog> {
  final _name = TextEditingController();
  final _bodyLocation = TextEditingController();

  @override
  void dispose() {
    _name.dispose();
    _bodyLocation.dispose();
    super.dispose();
  }

  void _submit() {
    final name = _name.text.trim();
    if (name.isEmpty) {
      return;
    }
    final location = _bodyLocation.text.trim();
    Navigator.of(context).pop(
      Lesion(
        name: name,
        bodyLocation: location.isEmpty ? null : location,
        createdAt: DateTime.now(),
      ),
    );
  }

  @override
  Widget build(BuildContext context) {
    return AlertDialog(
      title: const Text('New lesion'),
      content: Column(
        mainAxisSize: MainAxisSize.min,
        children: [
          TextField(
            controller: _name,
            autofocus: true,
            textCapitalization: TextCapitalization.sentences,
            textInputAction: TextInputAction.next,
            decoration: const InputDecoration(labelText: 'Name'),
          ),
          TextField(
            controller: _bodyLocation,
            textCapitalization: TextCapitalization.sentences,
            textInputAction: TextInputAction.done,
            onSubmitted: (_) => _submit(),
            decoration: const InputDecoration(
              labelText: 'Body location (optional)',
            ),
          ),
        ],
      ),
      actions: [
        TextButton(
          onPressed: () => Navigator.of(context).pop(),
          child: const Text('Cancel'),
        ),
        // Enabled only once a name has been typed.
        ValueListenableBuilder(
          valueListenable: _name,
          builder: (context, value, _) => FilledButton(
            onPressed: value.text.trim().isEmpty ? null : _submit,
            child: const Text('Create'),
          ),
        ),
      ],
    );
  }
}
