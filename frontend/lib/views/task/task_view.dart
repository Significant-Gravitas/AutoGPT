import 'package:auto_gpt_flutter_client/models/task.dart';
import 'package:auto_gpt_flutter_client/models/test_suite.dart';
import 'package:auto_gpt_flutter_client/viewmodels/settings_viewmodel.dart';
import 'package:auto_gpt_flutter_client/views/task/test_suite_detail_view.dart';
import 'package:auto_gpt_flutter_client/views/task/test_suite_list_tile.dart';
import 'package:flutter/material.dart';
import 'package:auto_gpt_flutter_client/viewmodels/task_viewmodel.dart';
import 'package:auto_gpt_flutter_client/viewmodels/chat_viewmodel.dart';
import 'package:auto_gpt_flutter_client/views/task/new_task_button.dart';
import 'package:auto_gpt_flutter_client/views/task/task_list_tile.dart';
import 'package:provider/provider.dart';

class TaskView extends StatefulWidget {
  final TaskViewModel viewModel;

  const TaskView({Key? key, required this.viewModel}) : super(key: key);

  @override
  _TaskViewState createState() => _TaskViewState();
}

class _TaskViewState extends State<TaskView> {
  @override
  void initState() {
    super.initState();

    // Schedule the fetchTasks call for after the initial build
    WidgetsBinding.instance.addPostFrameCallback((_) {
      widget.viewModel.fetchAndCombineData();
    });
  }

  @override
  Widget build(BuildContext context) {
    // Combine tasks and test suites into a single list
    final items = Provider.of<SettingsViewModel>(context, listen: false)
            .isDeveloperModeEnabled
        ? widget.viewModel.combinedDataSource
        : widget.viewModel.tasksDataSource;
    return Scaffold(
      backgroundColor: Colors.white,
      body: Stack(
        children: [
          Column(
            children: [
              // Title and New Task button
              Padding(
                  padding: const EdgeInsets.all(8.0),
                  child: NewTaskButton(
                    onPressed: () async {
                      // Update the current task ID and chats in ChatViewModel
                      final chatViewModel =
                          Provider.of<ChatViewModel>(context, listen: false);
                      chatViewModel.clearCurrentTaskAndChats();
                      widget.viewModel.deselectTask();
                      print(
                          'New Task button pressed, cleared current task ID and chats');
                    },
                  )),
              // Task List
              Expanded(
                child: ListView.builder(
                  itemCount: items.length,
                  itemBuilder: (context, index) {
                    final item = items[index];

                    if (item is Task) {
                      return TaskListTile(
                        task: item,
                        onTap: () {
                          // Select the task in TaskViewModel
                          widget.viewModel.selectTask(item.id);

                          // Update the current task ID in ChatViewModel
                          // TODO: Do we want to have a reference to chat view model in this class?
                          final chatViewModel = Provider.of<ChatViewModel>(
                              context,
                              listen: false);
                          chatViewModel.setCurrentTaskId(item.id);

                          print('Task ${item.title} tapped');
                        },
                        onDelete: () {
                          // Delete the task in TaskViewModel
                          widget.viewModel.deleteTask(item.id);
                          // TODO: Do we want to have a reference to chat view model in this class?
                          final chatViewModel = Provider.of<ChatViewModel>(
                              context,
                              listen: false);
                          if (chatViewModel.currentTaskId == item.id) {
                            chatViewModel.clearCurrentTaskAndChats();
                          }

                          print('Task ${item.title} delete button tapped');
                        },
                        selected: item.id == widget.viewModel.selectedTask?.id,
                      );
                    } else if (item is TestSuite) {
                      return TestSuiteListTile(
                        testSuite: item,
                        onTap: () {
                          // Navigate to the new view for this test suite
                          widget.viewModel.deselectTask();
                          widget.viewModel.selectTestSuite(item);
                          // TODO: Do we want to have a reference to chat view model in this class?
                          Provider.of<ChatViewModel>(context, listen: false)
                              .clearCurrentTaskAndChats();
                        },
                      );
                    } else {
                      return const SizedBox
                          .shrink(); // return an empty widget if type is unknown
                    }
                  },
                ),
              ),
            ],
          ),
          if (widget.viewModel.selectedTestSuite != null)
            Positioned(
              top: 0,
              left: 0,
              right: 0,
              bottom: 0,
              child: TestSuiteDetailView(
                testSuite: widget.viewModel.selectedTestSuite!,
                viewModel: widget.viewModel,
              ),
            ),
        ],
      ),
    );
  }
}
