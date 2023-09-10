import 'package:auto_gpt_flutter_client/viewmodels/api_settings_viewmodel.dart';
import 'package:auto_gpt_flutter_client/viewmodels/skill_tree_viewmodel.dart';
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
  runApp(
    MultiProvider(
      providers: [
        Provider(
          create: (context) => RestApiUtility("http://127.0.0.1:8000"),
        ),
        ProxyProvider<RestApiUtility, ChatService>(
          update: (context, restApiUtility, chatService) =>
              ChatService(restApiUtility),
        ),
        ProxyProvider<RestApiUtility, TaskService>(
          update: (context, restApiUtility, taskService) =>
              TaskService(restApiUtility),
        ),
        ChangeNotifierProxyProvider<RestApiUtility, ApiSettingsViewModel>(
          create: (context) => ApiSettingsViewModel(
              Provider.of<RestApiUtility>(context, listen: false)),
          update: (context, restApiUtility, apiSettingsViewModel) =>
              ApiSettingsViewModel(restApiUtility),
        ),
      ],
      child: MyApp(),
    ),
  );
}

class MyApp extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    // Fetch services from providers
    final chatService = Provider.of<ChatService>(context, listen: false);
    final taskService = Provider.of<TaskService>(context, listen: false);
    taskService.loadDeletedTasks();

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
          ChangeNotifierProvider(
            create: (context) => SkillTreeViewModel(),
          ),
        ],
        child: MainLayout(),
      ),
    );
  }
}
