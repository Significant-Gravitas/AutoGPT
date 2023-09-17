import 'package:auto_gpt_flutter_client/models/benchmark_service/report_request_body.dart';
import 'package:flutter/material.dart';
import 'package:auto_gpt_flutter_client/viewmodels/skill_tree_viewmodel.dart';
import 'package:provider/provider.dart';

// TODO: Add view model for task queue instead of skill tree view model
class TaskQueueView extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    final viewModel = Provider.of<SkillTreeViewModel>(context);

    // Reverse the node hierarchy
    final reversedHierarchy =
        viewModel.selectedNodeHierarchy?.reversed.toList() ?? [];

    // Convert reversedHierarchy to a list of test names
    final List<String> testNames =
        reversedHierarchy.map((node) => node.data.name).toList();

    return Material(
      color: Colors.white,
      child: Stack(
        children: [
          // The list of tasks (tiles)
          ListView.builder(
            itemCount: reversedHierarchy.length,
            itemBuilder: (context, index) {
              final node = reversedHierarchy[index];
              return Container(
                margin: EdgeInsets.fromLTRB(20, 5, 20, 5),
                decoration: BoxDecoration(
                  color: Colors.white, // white background
                  border: Border.all(
                      color: Colors.black, width: 1), // thin black border
                  borderRadius: BorderRadius.circular(4), // small corner radius
                ),
                child: ListTile(
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
                        // Create a ReportRequestBody with hardcoded values
                        ReportRequestBody reportRequestBody = ReportRequestBody(
                          category: "",
                          tests: testNames,
                          mock: true,
                        );

                        // Call runBenchmark method from SkillTreeViewModel
                        viewModel.runBenchmark(reportRequestBody);
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
