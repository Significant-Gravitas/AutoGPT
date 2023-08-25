import 'package:flutter/material.dart';
import 'views/main_layout.dart';
import 'package:provider/provider.dart';
import 'package:auto_gpt_flutter_client/viewmodels/task_viewmodel.dart';
import 'package:auto_gpt_flutter_client/viewmodels/chat_viewmodel.dart';

void main() {
  runApp(const MyApp());
}

class MyApp extends StatelessWidget {
  const MyApp({Key? key}) : super(key: key);

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'AutoGPT Flutter Client',
      theme: ThemeData(
        primarySwatch: Colors.blue,
      ),
      home: MultiProvider(
        providers: [
          ChangeNotifierProvider(create: (context) => TaskViewModel()),
          ChangeNotifierProvider(create: (context) => ChatViewModel()),
        ],
        child: const MainLayout(),
      ), // Set MainLayout as the home screen of the app
    );
  }
}
