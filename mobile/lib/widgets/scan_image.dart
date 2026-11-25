import 'dart:io';

import 'package:flutter/material.dart';

/// A scan's photo. Shows a placeholder icon instead if the file is missing or
/// can't be decoded. With no [width] or [height] it fills its constraints.
class ScanImage extends StatelessWidget {
  final String path;
  final double? width;
  final double? height;

  const ScanImage({super.key, required this.path, this.width, this.height});

  @override
  Widget build(BuildContext context) {
    final colors = Theme.of(context).colorScheme;

    return Image.file(
      File(path),
      width: width,
      height: height,
      fit: BoxFit.cover,
      errorBuilder: (context, error, stackTrace) => Container(
        width: width,
        height: height,
        alignment: Alignment.center,
        color: colors.surfaceContainerHighest,
        child: Icon(
          Icons.broken_image_outlined,
          color: colors.onSurfaceVariant,
        ),
      ),
    );
  }
}
