import 'package:flutter/material.dart';
import 'package:provider/provider.dart';

import 'screens/lesion_list_screen.dart';
import 'services/database_service.dart';
import 'services/inference_service.dart';
import 'services/tflite_inference_service.dart';

void main() {
  runApp(
    // One shared DatabaseService: sqflite reuses a single connection per
    // file, so closing a per-screen instance would close it for every screen.
    Provider<DatabaseService>(
      create: (_) => DatabaseService(),
      dispose: (_, db) => db.close(),
      // One shared model ensemble, loaded and warmed up once at startup
      // (lazy: false) rather than every time the camera screen opens.
      child: Provider<InferenceService>(
        lazy: false,
        create: (_) {
          final service = TFLiteInferenceService();
          // The camera screen awaits load() itself and shows any error, so
          // a failure here needn't be reported twice.
          service.load().ignore();
          return service;
        },
        dispose: (_, service) => service.dispose(),
        child: const MyApp(),
      ),
    ),
  );
}

class MyApp extends StatelessWidget {
  const MyApp({super.key});

  // This widget is the root of your application.
  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'EpiTrack',
      debugShowCheckedModeBanner: false,
      theme: ThemeData(
        // This is the theme of your application.
        //
        // TRY THIS: Try running your application with "flutter run". You'll see
        // the application has a purple toolbar. Then, without quitting the app,
        // try changing the seedColor in the colorScheme below to Colors.green
        // and then invoke "hot reload" (save your changes or press the "hot
        // reload" button in a Flutter-supported IDE, or press "r" if you used
        // the command line to start the app).
        //
        // Notice that the counter didn't reset back to zero; the application
        // state is not lost during the reload. To reset the state, use hot
        // restart instead.
        //
        // This works for code too, not just values: Most code changes can be
        // tested with just a hot reload.
        colorScheme: .fromSeed(seedColor: Colors.deepPurple),
      ),
      home: const LesionListScreen(),
    );
  }
}
