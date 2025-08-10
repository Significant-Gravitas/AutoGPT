import 'package:auto_gpt_flutter_client/models/test_suite.dart';
import 'package:auto_gpt_flutter_client/viewmodels/chat_viewmodel.dart';
import 'package:auto_gpt_flutter_client/viewmodels/task_viewmodel.dart';
import 'package:auto_gpt_flutter_client/views/task/task_list_tile.dart';
import 'package:flutter/material.dart';
import 'package:provider/provider.dart';

// TODO: Do we want a view model for every view?
class TestSuiteDetailView extends StatefulWidget {
  final TaskViewModel viewModel;
  final TestSuite testSuite;

  const TestSuiteDetailView(
      {Key? key, required this.testSuite, required this.viewModel})
      : super(key: key);

  @override
  _TestSuiteDetailViewState createState() => _TestSuiteDetailViewState();
}

class _TestSuiteDetailViewState extends State<TestSuiteDetailView> {
  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: Colors.white,
      appBar: AppBar(
        backgroundColor: Colors.grey,
        foregroundColor: Colors.black,
        title: Text("${widget.testSuite.timestamp}"),
        leading: IconButton(
          icon: Icon(Icons.arrow_back),
          onPressed: () => widget.viewModel.deselectTestSuite(),
        ),
      ),
      body: Column(
        children: [
          // Task List
          Expanded(
            child: ListView.builder(
              itemCount:
                  widget.testSuite.tests.length, // Count of tasks passed in
              itemBuilder: (context, index) {
                final task = widget.testSuite.tests[index];
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
                  selected: task.id == widget.viewModel.selectedTask?.id,
                );
              },
            ),
          ),
        ],
      ),
    );
  }
}
