import 'dart:io';

/// Deletes the image at [path]. A file that's already missing isn't an error,
/// and other file-system failures are ignored too: a leftover file is
/// harmless, but failing partway through a delete is not.
Future<void> deleteImageFile(String path) async {
  try {
    await File(path).delete();
  } on FileSystemException {
    // Nothing left to clean up.
  }
}
