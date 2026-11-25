class Lesion {
  final int? id;
  final String name;
  final String? bodyLocation;
  final DateTime createdAt;

  const Lesion({
    this.id,
    required this.name,
    this.bodyLocation,
    required this.createdAt,
  });

  factory Lesion.fromMap(Map<String, Object?> map) => Lesion(
    id: map['id'] as int?,
    name: map['name'] as String,
    bodyLocation: map['body_location'] as String?,
    createdAt: DateTime.fromMillisecondsSinceEpoch(map['created_at'] as int),
  );

  Map<String, Object?> toMap() => {
    if (id != null) 'id': id,
    'name': name,
    'body_location': bodyLocation,
    'created_at': createdAt.millisecondsSinceEpoch,
  };
}
