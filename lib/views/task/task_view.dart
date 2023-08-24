import 'package:auto_gpt_flutter_client/views/task/new_task_button.dart';
import 'package:auto_gpt_flutter_client/views/task/task_list_tile.dart';
import 'package:flutter/material.dart';
import 'package:auto_gpt_flutter_client/viewmodels/task_viewmodel.dart';

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
                  onPressed: () {
                    // TODO: Implement add new task action
                    print('Add new task button pressed');
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
                    // TODO: Implement the action when a task is tapped. This should trigger the TaskView to update.
                    print('Task ${task.title} tapped');
                  },
                  onDelete: () {
                    // TODO: Implement the action when a task is needing to be deleted. This should trigger the TaskView to update.
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
