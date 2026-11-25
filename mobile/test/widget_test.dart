import 'package:epitrack/main.dart';
import 'package:epitrack/models/lesion.dart';
import 'package:epitrack/services/database_service.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:provider/provider.dart';

/// No sqflite in widget tests, so the home screen gets an empty list.
class _FakeDatabaseService extends DatabaseService {
  @override
  Future<List<Lesion>> getLesions() async => [];
}

void main() {
  testWidgets('App builds and shows the empty lesion list', (tester) async {
    await tester.pumpWidget(
      Provider<DatabaseService>(
        create: (_) => _FakeDatabaseService(),
        child: const MyApp(),
      ),
    );
    await tester.pump();

    expect(find.text('EpiTrack'), findsOneWidget);
    expect(find.textContaining('No lesions tracked yet'), findsOneWidget);
  });
}
