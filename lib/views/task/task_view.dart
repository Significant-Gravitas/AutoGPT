import 'package:auto_gpt_flutter_client/viewmodels/chat_viewmodel.dart';
import 'package:auto_gpt_flutter_client/views/task/new_task_button.dart';
import 'package:auto_gpt_flutter_client/views/task/task_list_tile.dart';
import 'package:flutter/material.dart';
import 'package:auto_gpt_flutter_client/viewmodels/task_viewmodel.dart';
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
      widget.viewModel.fetchTasks();
    });
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: Colors.white,
      body: Column(
        children: [
          // Title and New Task button
          Padding(
            padding: const EdgeInsets.all(8.0),
            child: Column(
              children: [
                const Text(
                  'Tasks',
                  style: TextStyle(fontSize: 20, fontWeight: FontWeight.normal),
                ),
                const SizedBox(height: 8),
                NewTaskButton(
                  onPressed: () async {
                    // Update the current task ID and chats in ChatViewModel
                    final chatViewModel =
                        Provider.of<ChatViewModel>(context, listen: false);
                    chatViewModel.clearCurrentTaskAndChats();
                    print(
                        'New Task button pressed, cleared current task ID and chats');
                  },
                )
              ],
            ),
          ),
          // Task List
          Expanded(
            child: ListView.builder(
              itemCount: widget.viewModel.tasks.length,
              itemBuilder: (context, index) {
                final task = widget.viewModel.tasks[index];
                return TaskListTile(
                  task: task,
                  onTap: () {
                    // Select the task in TaskViewModel
                    widget.viewModel.selectTask(task.id);

                    // Update the current task ID in ChatViewModel
                    // TODO: Do we want to have a reference to chat view model in this class?
                    final chatViewModel =
                        Provider.of<ChatViewModel>(context, listen: false);
                    chatViewModel.setCurrentTaskId(task.id);

                    print('Task ${task.title} tapped');
                  },
                  onDelete: () {
                    // Delete the task in TaskViewModel
                    widget.viewModel.deleteTask(task.id);
                    // TODO: Do we want to have a reference to chat view model in this class?
                    final chatViewModel =
                        Provider.of<ChatViewModel>(context, listen: false);
                    if (chatViewModel.currentTaskId == task.id) {
                      chatViewModel.clearCurrentTaskAndChats();
                    }

                    print('Task ${task.title} delete button tapped');
                  },
                );
              },
            ),
          ),
        ],
      ),
    );
  }
}
