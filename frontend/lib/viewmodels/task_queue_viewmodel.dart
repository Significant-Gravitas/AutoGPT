import 'package:auto_gpt_flutter_client/models/benchmark/benchmark_run.dart';
import 'package:auto_gpt_flutter_client/models/benchmark/benchmark_step_request_body.dart';
import 'package:auto_gpt_flutter_client/models/benchmark/benchmark_task_request_body.dart';
import 'package:auto_gpt_flutter_client/models/benchmark/benchmark_task_status.dart';
import 'package:auto_gpt_flutter_client/models/skill_tree/skill_tree_edge.dart';
import 'package:auto_gpt_flutter_client/models/skill_tree/skill_tree_node.dart';
import 'package:auto_gpt_flutter_client/models/step.dart';
import 'package:auto_gpt_flutter_client/models/task.dart';
import 'package:auto_gpt_flutter_client/models/test_option.dart';
import 'package:auto_gpt_flutter_client/models/test_suite.dart';
import 'package:auto_gpt_flutter_client/services/benchmark_service.dart';
import 'package:auto_gpt_flutter_client/services/leaderboard_service.dart';
import 'package:auto_gpt_flutter_client/services/shared_preferences_service.dart';
import 'package:auto_gpt_flutter_client/viewmodels/chat_viewmodel.dart';
import 'package:auto_gpt_flutter_client/viewmodels/task_viewmodel.dart';
import 'package:collection/collection.dart';
import 'package:flutter/foundation.dart';
import 'package:uuid/uuid.dart';
import 'package:auto_gpt_flutter_client/utils/stack.dart';

class TaskQueueViewModel extends ChangeNotifier {
  final BenchmarkService benchmarkService;
  final LeaderboardService leaderboardService;
  final SharedPreferencesService prefsService;
  bool isBenchmarkRunning = false;
  Map<SkillTreeNode, BenchmarkTaskStatus> benchmarkStatusMap = {};
  List<BenchmarkRun> currentBenchmarkRuns = [];
  List<SkillTreeNode>? _selectedNodeHierarchy;
  TestOption _selectedOption = TestOption.runSingleTest;

  TestOption get selectedOption => _selectedOption;
  List<SkillTreeNode>? get selectedNodeHierarchy => _selectedNodeHierarchy;

  TaskQueueViewModel(
      this.benchmarkService, this.leaderboardService, this.prefsService);

  void updateSelectedNodeHierarchyBasedOnOption(
      TestOption selectedOption,
      SkillTreeNode? selectedNode,
      List<SkillTreeNode> nodes,
      List<SkillTreeEdge> edges) {
    _selectedOption = selectedOption;
    switch (selectedOption) {
      case TestOption.runSingleTest:
        _selectedNodeHierarchy = selectedNode != null ? [selectedNode] : [];
        break;

      case TestOption.runTestSuiteIncludingSelectedNodeAndAncestors:
        if (selectedNode != null) {
          populateSelectedNodeHierarchy(selectedNode.id, nodes, edges);
        }
        break;

      case TestOption.runAllTestsInCategory:
        if (selectedNode != null) {
          _getAllNodesInDepthFirstOrderEnsuringParents(nodes, edges);
        }
        break;
    }
    notifyListeners();
  }

  void _getAllNodesInDepthFirstOrderEnsuringParents(
      List<SkillTreeNode> skillTreeNodes, List<SkillTreeEdge> skillTreeEdges) {
    var nodes = <SkillTreeNode>[];
    var stack = Stack<SkillTreeNode>();
    var visited = <String>{};

    // Identify the root node by its label
    var root = skillTreeNodes.firstWhere((node) => node.label == "WriteFile");

    stack.push(root);
    visited.add(root.id);

    while (stack.isNotEmpty) {
      var node = stack.peek(); // Peek the top node, but do not remove it yet
      var parents =
          _getParentsOfNodeUsingEdges(node.id, skillTreeNodes, skillTreeEdges);

      // Check if all parents are visited
      if (parents.every((parent) => visited.contains(parent.id))) {
        nodes.add(node);
        stack.pop(); // Remove the node only when all its parents are visited

        // Get the children of the current node using edges
        var children = _getChildrenOfNodeUsingEdges(
                node.id, skillTreeNodes, skillTreeEdges)
            .where((child) => !visited.contains(child.id));

        children.forEach((child) {
          visited.add(child.id);
          stack.push(child);
        });
      } else {
        stack
            .pop(); // Remove the node if not all parents are visited, it will be re-added when its parents are visited
      }
    }

    _selectedNodeHierarchy = nodes;
  }

  List<SkillTreeNode> _getParentsOfNodeUsingEdges(
      String nodeId, List<SkillTreeNode> nodes, List<SkillTreeEdge> edges) {
    var parents = <SkillTreeNode>[];

    for (var edge in edges) {
      if (edge.to == nodeId) {
        parents.add(nodes.firstWhere((node) => node.id == edge.from));
      }
    }

    return parents;
  }

  List<SkillTreeNode> _getChildrenOfNodeUsingEdges(
      String nodeId, List<SkillTreeNode> nodes, List<SkillTreeEdge> edges) {
    var children = <SkillTreeNode>[];

    for (var edge in edges) {
      if (edge.from == nodeId) {
        children.add(nodes.firstWhere((node) => node.id == edge.to));
      }
    }

    return children;
  }

  // TODO: Do we want to continue testing other branches of tree if one branch side fails benchmarking?
  void populateSelectedNodeHierarchy(String startNodeId,
      List<SkillTreeNode> nodes, List<SkillTreeEdge> edges) {
    _selectedNodeHierarchy = <SkillTreeNode>[];
    final addedNodes = <String>{};
    recursivePopulateHierarchy(startNodeId, addedNodes, nodes, edges);
    notifyListeners();
  }

  void recursivePopulateHierarchy(String nodeId, Set<String> addedNodes,
      List<SkillTreeNode> nodes, List<SkillTreeEdge> edges) {
    // Find the current node in the skill tree nodes list.
    final currentNode = nodes.firstWhereOrNull((node) => node.id == nodeId);

    // If the node is found and it hasn't been added yet, proceed with the population.
    if (currentNode != null && addedNodes.add(currentNode.id)) {
      // Find all parent edges for the current node.
      final parentEdges = edges.where((edge) => edge.to == currentNode.id);

      // For each parent edge found, recurse to the parent node.
      for (final parentEdge in parentEdges) {
        // Recurse to the parent node identified by the 'from' field of the edge.
        recursivePopulateHierarchy(parentEdge.from, addedNodes, nodes, edges);
      }

      // After processing all parent nodes, add the current node to the list.
      _selectedNodeHierarchy!.add(currentNode);
    }
  }

  Future<void> runBenchmark(
      ChatViewModel chatViewModel, TaskViewModel taskViewModel) async {
    // Clear the benchmarkStatusList
    benchmarkStatusMap.clear();

    // Reset the current benchmark runs list to be empty at the start of a new benchmark
    currentBenchmarkRuns = [];

    // Create a new TestSuite object with the current timestamp
    final testSuite =
        TestSuite(timestamp: DateTime.now().toIso8601String(), tests: []);

    // Set the benchmark running flag to true
    isBenchmarkRunning = true;
    // Notify listeners
    notifyListeners();

    // Populate benchmarkStatusList with node hierarchy
    for (var node in _selectedNodeHierarchy!) {
      benchmarkStatusMap[node] = BenchmarkTaskStatus.notStarted;
    }

    try {
      // Loop through the nodes in the hierarchy
      for (var node in _selectedNodeHierarchy!) {
        benchmarkStatusMap[node] = BenchmarkTaskStatus.inProgress;
        notifyListeners();

        // Create a BenchmarkTaskRequestBody
        final benchmarkTaskRequestBody = BenchmarkTaskRequestBody(
            input: node.data.task, evalId: node.data.evalId);

        // Create a new benchmark task
        final createdTask = await benchmarkService
            .createBenchmarkTask(benchmarkTaskRequestBody);

        // Create a new Task object
        final task =
            Task(id: createdTask['task_id'], title: createdTask['input']);

        // Update the current task ID in ChatViewModel
        chatViewModel.setCurrentTaskId(task.id);

        // Execute the first step and initialize the Step object
        Map<String, dynamic> stepResponse =
            await benchmarkService.executeBenchmarkStep(
                task.id, BenchmarkStepRequestBody(input: node.data.task));
        Step step = Step.fromMap(stepResponse);
        chatViewModel.fetchChatsForTask();

        // Check if it's the last step
        while (!step.isLast) {
          // Execute next step and update the Step object
          stepResponse = await benchmarkService.executeBenchmarkStep(
              task.id, BenchmarkStepRequestBody(input: null));
          step = Step.fromMap(stepResponse);

          // Fetch chats for the task
          chatViewModel.fetchChatsForTask();
        }

        // Trigger the evaluation
        final evaluationResponse =
            await benchmarkService.triggerEvaluation(task.id);

        // Decode the evaluationResponse into a BenchmarkRun object
        BenchmarkRun benchmarkRun = BenchmarkRun.fromJson(evaluationResponse);

        // Add the benchmark run object to the list of current benchmark runs
        currentBenchmarkRuns.add(benchmarkRun);

        // Update the benchmarkStatusList based on the evaluation response
        bool successStatus = benchmarkRun.metrics.success;
        benchmarkStatusMap[node] = successStatus
            ? BenchmarkTaskStatus.success
            : BenchmarkTaskStatus.failure;
        await Future.delayed(Duration(seconds: 1));
        notifyListeners();

        testSuite.tests.add(task);
        // If successStatus is false, break out of the loop
        if (!successStatus) {
          print(
              "Benchmark for node ${node.id} failed. Stopping all benchmarks.");
          break;
        }
      }

      // Add the TestSuite to the TaskViewModel
      taskViewModel.addTestSuite(testSuite);
    } catch (e) {
      print("Error while running benchmark: $e");
    }

    // Reset the benchmark running flag
    isBenchmarkRunning = false;
    notifyListeners();
  }

  Future<void> submitToLeaderboard(
      String teamName, String repoUrl, String agentGitCommitSha) async {
    // Create a UUID.v4 for our unique run ID
    String uuid = const Uuid().v4();

    for (var run in currentBenchmarkRuns) {
      run.repositoryInfo.teamName = teamName;
      run.repositoryInfo.repoUrl = repoUrl;
      run.repositoryInfo.agentGitCommitSha = agentGitCommitSha;
      run.runDetails.runId = uuid;

      await leaderboardService.submitReport(run);
      print('Completed submission to leaderboard!');
    }

    // Clear the currentBenchmarkRuns list after submitting to the leaderboard
    currentBenchmarkRuns.clear();
  }
}
