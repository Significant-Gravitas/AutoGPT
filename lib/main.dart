import 'package:auto_gpt_flutter_client/viewmodels/api_settings_viewmodel.dart';
import 'package:flutter/material.dart';
import 'views/main_layout.dart';
import 'package:provider/provider.dart';
import 'package:auto_gpt_flutter_client/viewmodels/task_viewmodel.dart';
import 'package:auto_gpt_flutter_client/viewmodels/chat_viewmodel.dart';
import 'package:auto_gpt_flutter_client/services/chat_service.dart';
import 'package:auto_gpt_flutter_client/services/task_service.dart';
import 'package:auto_gpt_flutter_client/utils/rest_api_utility.dart';

// TODO: Update documentation throughout project for consistency
void main() {
  // Initialize the RestApiUtility
  final restApiUtility = RestApiUtility("http://127.0.0.1:8000");

  // Initialize the services
  final chatService = ChatService(restApiUtility);
  final taskService = TaskService(restApiUtility);

  runApp(MyApp(
    chatService: chatService,
    taskService: taskService,
  ));
}

class MyApp extends StatelessWidget {
  final ChatService chatService;
  final TaskService taskService;

  const MyApp({Key? key, required this.chatService, required this.taskService})
      : super(key: key);

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'AutoGPT Flutter Client',
      theme: ThemeData(
        primarySwatch: Colors.blue,
      ),
      home: MultiProvider(
        providers: [
          ChangeNotifierProvider(
              create: (context) => ChatViewModel(chatService)),
          ChangeNotifierProvider(
              create: (context) => TaskViewModel(taskService)),
          ChangeNotifierProvider(create: (context) => ApiSettingsViewModel()),
        ],
        child: const MainLayout(),
      ), // Set MainLayout as the home screen of the app
    );
  }
}
