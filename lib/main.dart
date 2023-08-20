import 'package:flutter/material.dart';
import 'views/main_layout.dart';

void main() {
  runApp(const MyApp());
}

class MyApp extends StatelessWidget {
  const MyApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'AutoGPT Flutter Client',
      theme: ThemeData(
        primarySwatch: Colors.blue,
      ),
      home: const MainLayout(), // Set MainLayout as the home screen of the app
    );
  }
}
