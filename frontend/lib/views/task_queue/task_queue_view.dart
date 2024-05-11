import 'package:auto_gpt_flutter_client/models/benchmark/benchmark_task_status.dart';
import 'package:auto_gpt_flutter_client/models/test_option.dart';
import 'package:auto_gpt_flutter_client/viewmodels/chat_viewmodel.dart';
import 'package:auto_gpt_flutter_client/viewmodels/skill_tree_viewmodel.dart';
import 'package:auto_gpt_flutter_client/viewmodels/task_queue_viewmodel.dart';
import 'package:auto_gpt_flutter_client/viewmodels/task_viewmodel.dart';
import 'package:auto_gpt_flutter_client/views/task_queue/test_suite_button.dart';
import 'package:flutter/material.dart';
import 'package:provider/provider.dart';

class TaskQueueView extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    // TODO: This should be injected instead
    final viewModel = Provider.of<TaskQueueViewModel>(context);

    // Node hierarchy
    final nodeHierarchy = viewModel.selectedNodeHierarchy ?? [];

    return Material(
      color: Colors.white,
      child: Column(
        children: [
          // The list of tasks (tiles)
          Expanded(
            child: ListView.builder(
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
          ),

          // Buttons at the bottom
          Padding(
            padding: EdgeInsets.all(20),
            child: Column(
              children: [
                // TestSuiteButton
                TestSuiteButton(
                  isDisabled: viewModel.isBenchmarkRunning,
                  selectedOptionString: viewModel.selectedOption.description,
                  onOptionSelected: (selectedOption) {
                    print('Option Selected: $selectedOption');
                    final skillTreeViewModel =
                        Provider.of<SkillTreeViewModel>(context, listen: false);
                    viewModel.updateSelectedNodeHierarchyBasedOnOption(
                        TestOptionExtension.fromDescription(selectedOption)!,
                        skillTreeViewModel.selectedNode,
                        skillTreeViewModel.skillTreeNodes,
                        skillTreeViewModel.skillTreeEdges);
                  },
                  onPlayPressed: (selectedOption) {
                    print('Starting benchmark with option: $selectedOption');
                    final chatViewModel =
                        Provider.of<ChatViewModel>(context, listen: false);
                    final taskViewModel =
                        Provider.of<TaskViewModel>(context, listen: false);
                    chatViewModel.clearCurrentTaskAndChats();
                    viewModel.runBenchmark(chatViewModel, taskViewModel);
                  },
                ),
                SizedBox(height: 8), // Gap of 8 points between buttons
              ],
            ),
          ),
        ],
      ),
    );
  }
}
