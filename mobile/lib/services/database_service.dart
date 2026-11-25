import 'package:path/path.dart' as p;
import 'package:sqflite/sqflite.dart';

import '../models/lesion.dart';
import '../models/scan.dart';

class DatabaseService {
  Future<Database>? _database;

  Future<Database> get _db => _database ??= _open();

  Future<Database> _open() async {
    final path = p.join(await getDatabasesPath(), 'epitrack.db');
    return openDatabase(
      path,
      version: 1,
      onConfigure: (db) => db.execute('PRAGMA foreign_keys = ON'),
      onCreate: _onCreate,
    );
  }

  Future<void> _onCreate(Database db, int version) async {
    await db.execute('''
      CREATE TABLE lesions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT NOT NULL,
        body_location TEXT,
        created_at INTEGER NOT NULL
      )
    ''');
    await db.execute('''
      CREATE TABLE scans (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        lesion_id INTEGER NOT NULL,
        image_path TEXT NOT NULL,
        captured_at INTEGER NOT NULL,
        predictions TEXT NOT NULL,
        inference_time_ms INTEGER NOT NULL,
        note TEXT,
        FOREIGN KEY (lesion_id) REFERENCES lesions (id) ON DELETE CASCADE
      )
    ''');
    await db.execute('''
      CREATE INDEX idx_scans_lesion_captured
      ON scans (lesion_id, captured_at)
    ''');
  }

  Future<int> insertLesion(Lesion lesion) async {
    final db = await _db;
    return db.insert('lesions', lesion.toMap());
  }

  /// Newest first.
  Future<List<Lesion>> getLesions() async {
    final db = await _db;
    final rows = await db.query('lesions', orderBy: 'created_at DESC, id DESC');
    return rows.map(Lesion.fromMap).toList();
  }

  /// Also deletes the lesion's scans (ON DELETE CASCADE), but not their
  /// image files.
  Future<void> deleteLesion(int id) async {
    final db = await _db;
    await db.delete('lesions', where: 'id = ?', whereArgs: [id]);
  }

  Future<int> insertScan(Scan scan) async {
    final db = await _db;
    return db.insert('scans', scan.toMap());
  }

  /// Doesn't delete the scan's image file.
  Future<void> deleteScan(int id) async {
    final db = await _db;
    await db.delete('scans', where: 'id = ?', whereArgs: [id]);
  }

  /// Oldest first, so the list reads as a progression timeline.
  Future<List<Scan>> getScansForLesion(int lesionId) async {
    final db = await _db;
    final rows = await db.query(
      'scans',
      where: 'lesion_id = ?',
      whereArgs: [lesionId],
      orderBy: 'captured_at ASC, id ASC',
    );
    return rows.map(Scan.fromMap).toList();
  }

  /// The most recent scan of [lesionId], or null if it has none.
  Future<Scan?> getLatestScan(int lesionId) async {
    final db = await _db;
    final rows = await db.query(
      'scans',
      where: 'lesion_id = ?',
      whereArgs: [lesionId],
      orderBy: 'captured_at DESC, id DESC',
      limit: 1,
    );
    return rows.isEmpty ? null : Scan.fromMap(rows.first);
  }

  Future<void> close() async {
    final database = _database;
    _database = null;
    if (database != null) {
      await (await database).close();
    }
  }
}
