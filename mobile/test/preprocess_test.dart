import 'dart:io';

import 'package:epitrack/services/tflite_inference_service.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:image/image.dart' as img;

void main() {
  test('inputPngPathFor puts the PNG next to the scan', () {
    final sep = Platform.pathSeparator;
    expect(
      inputPngPathFor('${sep}scans${sep}scan_123.jpg'),
      '${sep}scans${sep}scan_123_input.png',
    );
  });

  test('saved input PNG has exactly the pixels the models get', () {
    final dir = Directory.systemTemp.createTempSync('epitrack_test');
    addTearDown(() => dir.deleteSync(recursive: true));

    // A non-square photo-sized image with a gradient, saved as a JPEG.
    final source = img.Image(width: 640, height: 480);
    for (final pixel in source) {
      pixel
        ..r = pixel.x * 255 ~/ 639
        ..g = pixel.y * 255 ~/ 479
        ..b = (pixel.x + pixel.y) % 256;
    }
    final jpg = '${dir.path}/scan_1.jpg';
    File(jpg).writeAsBytesSync(img.encodeJpg(source));

    final pngPath = inputPngPathFor(jpg);
    final input = preprocessImageFile(jpg, saveInputPngTo: pngPath);
    expect(input, hasLength(380 * 380 * 3));

    final png = img.decodePng(File(pngPath).readAsBytesSync())!;
    expect(png.width, 380);
    expect(png.height, 380);
    var i = 0;
    for (var y = 0; y < 380; y++) {
      for (var x = 0; x < 380; x++) {
        final pixel = png.getPixel(x, y);
        expect(
          [pixel.r, pixel.g, pixel.b],
          [input[i], input[i + 1], input[i + 2]],
          reason: 'pixel ($x, $y)',
        );
        i += 3;
      }
    }
  });
}
