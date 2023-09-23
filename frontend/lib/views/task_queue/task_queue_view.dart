import 'package:auto_gpt_flutter_client/models/benchmark/benchmark_task_status.dart';
import 'package:auto_gpt_flutter_client/viewmodels/chat_viewmodel.dart';
import 'package:auto_gpt_flutter_client/viewmodels/task_viewmodel.dart';
import 'package:flutter/material.dart';
import 'package:auto_gpt_flutter_client/viewmodels/skill_tree_viewmodel.dart';
import 'package:provider/provider.dart';

// TODO: Add view model for task queue instead of skill tree view model
class TaskQueueView extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    final viewModel = Provider.of<SkillTreeViewModel>(context);

    // Node hierarchy
    final nodeHierarchy = viewModel.selectedNodeHierarchy ?? [];

    return Material(
      color: Colors.white,
      child: Stack(
        children: [
          // The list of tasks (tiles)
          ListView.builder(
            itemCount: nodeHierarchy.length,
            itemBuilder: (context, index) {
              final node = nodeHierarchy[index];

              // Choose the appropriate leading widget based on the task status
              Widget leadingWidget;
              switch (viewModel.benchmarkStatusMap[node]) {
                case null:
                case BenchmarkTaskStatus.notStarted:
                  leadingWidget = CircleAvatar(
                    radius: 12,
                    backgroundColor: Colors.grey,
                    child: CircleAvatar(
                      radius: 6,
                      backgroundColor: Colors.white,
                    ),
                  );
                  break;
                case BenchmarkTaskStatus.inProgress:
                  leadingWidget = SizedBox(
                    width: 24,
                    height: 24,
                    child: CircularProgressIndicator(
                      strokeWidth: 2,
                    ),
                  );
                  break;
                case BenchmarkTaskStatus.success:
                  leadingWidget = CircleAvatar(
                    radius: 12,
                    backgroundColor: Colors.green,
                    child: CircleAvatar(
                      radius: 6,
                      backgroundColor: Colors.white,
                    ),
                  );
                  break;
                case BenchmarkTaskStatus.failure:
                  leadingWidget = CircleAvatar(
                    radius: 12,
                    backgroundColor: Colors.red,
                    child: CircleAvatar(
                      radius: 6,
                      backgroundColor: Colors.white,
                    ),
                  );
                  break;
              }

              return Container(
                margin: EdgeInsets.fromLTRB(20, 5, 20, 5),
                decoration: BoxDecoration(
                  color: Colors.white,
                  border: Border.all(color: Colors.black, width: 1),
                  borderRadius: BorderRadius.circular(4),
                ),
                child: ListTile(
                  leading: leadingWidget,
                  title: Center(child: Text('${node.label}')),
                  subtitle:
                      Center(child: Text('${node.data.info.description}')),
                ),
              );
            },
          ),

          Positioned(
            bottom: 20,
            left: 20,
            right: 20,
            child: Container(
              decoration: BoxDecoration(
                borderRadius: BorderRadius.circular(8.0),
                border: Border.all(color: Colors.green, width: 3),
              ),
              child: ElevatedButton(
                onPressed: viewModel.isBenchmarkRunning
                    ? null
                    : () {
                        // TODO: We should not be passing this dependency in like this
                        final chatViewModel =
                            Provider.of<ChatViewModel>(context, listen: false);
                        final taskViewModel =
                            Provider.of<TaskViewModel>(context, listen: false);
                        chatViewModel.clearCurrentTaskAndChats();
                        // Call runBenchmark method from SkillTreeViewModel
                        viewModel.runBenchmark(chatViewModel, taskViewModel);
                      },
                child: Row(
                  mainAxisAlignment:
                      MainAxisAlignment.center, // Center the children
                  children: [
                    Text(
                      'Initiate test suite',
                      style: TextStyle(
                        color: Colors.green, // Text color is set to green
                        fontWeight: FontWeight.bold, // Make text bold
                        fontSize: 16, // Increase font size
                      ),
                    ),
                    SizedBox(width: 10), // Gap of 10 between text and icon
                    Icon(
                      Icons.play_arrow,
                      color: Colors.green, // Icon color is set to green
                      size: 24, // Increase icon size
                    ),
                  ],
                ),
                style: ButtonStyle(
                  backgroundColor: MaterialStateProperty.all(Colors.white),
                  shape: MaterialStateProperty.all<RoundedRectangleBorder>(
                    RoundedRectangleBorder(
                      borderRadius: BorderRadius.circular(8.0),
                    ),
                  ),
                  minimumSize: MaterialStateProperty.all(
                      Size(double.infinity, 50)), // Full width
                  padding: MaterialStateProperty.all(EdgeInsets.all(0)),
                ),
              ),
            ),
          ),
        ],
      ),
    );
  }
}
