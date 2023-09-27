import 'package:auto_gpt_flutter_client/services/leaderboard_service.dart';
import 'package:auto_gpt_flutter_client/viewmodels/settings_viewmodel.dart';
import 'package:auto_gpt_flutter_client/views/auth/firebase_auth_view.dart';
import 'package:flutter/material.dart';
import 'views/main_layout.dart';
import 'package:provider/provider.dart';
import 'package:firebase_core/firebase_core.dart';
import 'package:firebase_auth/firebase_auth.dart';

import 'package:auto_gpt_flutter_client/viewmodels/task_viewmodel.dart';
import 'package:auto_gpt_flutter_client/viewmodels/chat_viewmodel.dart';
import 'package:auto_gpt_flutter_client/viewmodels/skill_tree_viewmodel.dart';

import 'package:auto_gpt_flutter_client/services/chat_service.dart';
import 'package:auto_gpt_flutter_client/services/task_service.dart';
import 'package:auto_gpt_flutter_client/services/benchmark_service.dart';

import 'package:auto_gpt_flutter_client/utils/rest_api_utility.dart';

// TODO: Update documentation throughout project for consistency
void main() async {
  WidgetsFlutterBinding.ensureInitialized();
  await Firebase.initializeApp(
    options: const FirebaseOptions(
      apiKey: 'AIzaSyBvYLAK_A0uhFuVPQbTxUdVWbb_Lsur9cg',
      authDomain: 'prod-auto-gpt.firebaseapp.com',
      projectId: 'prod-auto-gpt',
      storageBucket: 'prod-auto-gpt.appspot.com',
      messagingSenderId: '387936576242',
      appId: '1:387936576242:web:7536e0c50dd81b4dd7a66b',
      measurementId: 'G-8PRS69JJRL',
    ),
  );

  runApp(
    MultiProvider(
      providers: [
        Provider(
          create: (context) => RestApiUtility("http://127.0.0.1:8000/ap/v1"),
        ),
        ProxyProvider<RestApiUtility, ChatService>(
          update: (context, restApiUtility, chatService) =>
              ChatService(restApiUtility),
        ),
        ProxyProvider<RestApiUtility, TaskService>(
          update: (context, restApiUtility, taskService) =>
              TaskService(restApiUtility),
        ),
        ProxyProvider<RestApiUtility, BenchmarkService>(
          update: (context, restApiUtility, benchmarkService) =>
              BenchmarkService(restApiUtility),
        ),
        ProxyProvider<RestApiUtility, LeaderboardService>(
          update: (context, restApiUtility, leaderboardService) =>
              LeaderboardService(restApiUtility),
        ),
        ChangeNotifierProxyProvider<RestApiUtility, SettingsViewModel>(
          create: (context) => SettingsViewModel(
              Provider.of<RestApiUtility>(context, listen: false)),
          update: (context, restApiUtility, settingsViewModel) =>
              SettingsViewModel(restApiUtility),
        ),
      ],
      child: MyApp(),
    ),
  );
}

class MyApp extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    final taskService = Provider.of<TaskService>(context, listen: false);
    taskService.loadDeletedTasks();

    return MaterialApp(
      title: 'AutoGPT Flutter Client',
      theme: ThemeData(
        primarySwatch: Colors.blue,
      ),
      home: StreamBuilder<User?>(
        stream: FirebaseAuth.instance.authStateChanges(),
        builder: (context, snapshot) {
          if (snapshot.connectionState == ConnectionState.waiting) {
            return CircularProgressIndicator();
          }
          if (snapshot.hasData && snapshot.data != null) {
            return MultiProvider(
              providers: [
                ChangeNotifierProvider(
                    create: (context) => ChatViewModel(
                        Provider.of<ChatService>(context, listen: false))),
                ChangeNotifierProvider(
                    create: (context) => TaskViewModel(
                        Provider.of<TaskService>(context, listen: false))),
                ChangeNotifierProvider(
                  create: (context) => SkillTreeViewModel(
                      Provider.of<BenchmarkService>(context, listen: false),
                      Provider.of<LeaderboardService>(context, listen: false)),
                ),
              ],
              child: MainLayout(),
            ); // User is signed in
          }
          return FirebaseAuthView(); // User is not signed in, show auth UI
        },
      ),
    );
  }
}
