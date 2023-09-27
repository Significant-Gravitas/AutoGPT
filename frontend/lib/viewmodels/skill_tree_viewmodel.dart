import 'dart:convert';
import 'package:auto_gpt_flutter_client/models/benchmark/benchmark_run.dart';
import 'package:auto_gpt_flutter_client/models/benchmark/benchmark_step_request_body.dart';
import 'package:auto_gpt_flutter_client/models/benchmark/benchmark_task_request_body.dart';
import 'package:auto_gpt_flutter_client/models/benchmark/benchmark_task_status.dart';
import 'package:auto_gpt_flutter_client/models/skill_tree/skill_tree_category.dart';
import 'package:auto_gpt_flutter_client/models/skill_tree/skill_tree_edge.dart';
import 'package:auto_gpt_flutter_client/models/skill_tree/skill_tree_node.dart';
import 'package:auto_gpt_flutter_client/models/step.dart';
import 'package:auto_gpt_flutter_client/models/task.dart';
import 'package:auto_gpt_flutter_client/models/test_suite.dart';
import 'package:auto_gpt_flutter_client/services/benchmark_service.dart';
import 'package:auto_gpt_flutter_client/services/leaderboard_service.dart';
import 'package:auto_gpt_flutter_client/viewmodels/chat_viewmodel.dart';
import 'package:auto_gpt_flutter_client/viewmodels/task_viewmodel.dart';
import 'package:collection/collection.dart';
import 'package:flutter/foundation.dart';
import 'package:flutter/services.dart';
import 'package:graphview/GraphView.dart';
import 'package:uuid/uuid.dart';

class SkillTreeViewModel extends ChangeNotifier {
  // TODO: Potentially move to task queue view model when we create one
  final BenchmarkService benchmarkService;
  // TODO: Potentially move to task queue view model when we create one
  final LeaderboardService leaderboardService;
  // TODO: Potentially move to task queue view model when we create one
  bool isBenchmarkRunning = false;
  // TODO: Potentially move to task queue view model when we create one
  // TODO: clear when clicking a new node
  Map<SkillTreeNode, BenchmarkTaskStatus> benchmarkStatusMap = {};

  List<BenchmarkRun> currentBenchmarkRuns = [];

  List<SkillTreeNode> _skillTreeNodes = [];
  List<SkillTreeEdge> _skillTreeEdges = [];
  SkillTreeNode? _selectedNode;
  // TODO: Potentially move to task queue view model when we create one
  List<SkillTreeNode>? _selectedNodeHierarchy;

  List<SkillTreeNode> get skillTreeNodes => _skillTreeNodes;
  List<SkillTreeEdge> get skillTreeEdges => _skillTreeEdges;
  SkillTreeNode? get selectedNode => _selectedNode;
  List<SkillTreeNode>? get selectedNodeHierarchy => _selectedNodeHierarchy;

  final Graph graph = Graph();
  SugiyamaConfiguration builder = SugiyamaConfiguration();

  SkillTreeCategory currentSkillTreeType = SkillTreeCategory.general;

  SkillTreeViewModel(this.benchmarkService, this.leaderboardService);

  Future<void> initializeSkillTree() async {
    try {
      resetState();

      String fileName = currentSkillTreeType.jsonFileName;

      // Read the JSON file from assets
      String jsonContent = await rootBundle.loadString('assets/$fileName');

      // Decode the JSON string
      Map<String, dynamic> decodedJson = jsonDecode(jsonContent);

      // Create SkillTreeNodes from the decoded JSON
      for (var nodeMap in decodedJson['nodes']) {
        SkillTreeNode node = SkillTreeNode.fromJson(nodeMap);
        _skillTreeNodes.add(node);
      }

      // Create SkillTreeEdges from the decoded JSON
      for (var edgeMap in decodedJson['edges']) {
        SkillTreeEdge edge = SkillTreeEdge.fromJson(edgeMap);
        _skillTreeEdges.add(edge);
      }

      builder.orientation = (SugiyamaConfiguration.ORIENTATION_LEFT_RIGHT);
      builder.bendPointShape = CurvedBendPointShape(curveLength: 20);

      notifyListeners();

      return Future.value(); // Explicitly return a completed Future
    } catch (e) {
      print(e);
    }
  }

  void resetState() {
    _skillTreeNodes = [];
    _skillTreeEdges = [];
    _selectedNode = null;
    _selectedNodeHierarchy = null;
  }

  void toggleNodeSelection(String nodeId) {
    if (isBenchmarkRunning) return;
    if (_selectedNode?.id == nodeId) {
      // Unselect the node if it's already selected
      _selectedNode = null;
      _selectedNodeHierarchy = null;
    } else {
      // Select the new node
      _selectedNode = _skillTreeNodes.firstWhere((node) => node.id == nodeId);
      populateSelectedNodeHierarchy(nodeId);
    }
    notifyListeners();
  }

  // TODO: Do we want to continue testing other branches of tree if one branch side fails benchmarking?
  void populateSelectedNodeHierarchy(String startNodeId) {
    // Initialize an empty list to hold the nodes in all hierarchies.
    _selectedNodeHierarchy = <SkillTreeNode>[];

    // Initialize a set to keep track of nodes that have been added.
    final addedNodes = <String>{};

    // Start the recursive population of the hierarchy from the startNodeId.
    recursivePopulateHierarchy(startNodeId, addedNodes);

    // Notify listeners about the change in the selectedNodeHierarchy state.
    notifyListeners();
  }

  void recursivePopulateHierarchy(String nodeId, Set<String> addedNodes) {
    // Find the current node in the skill tree nodes list.
    final currentNode =
        _skillTreeNodes.firstWhereOrNull((node) => node.id == nodeId);

    // If the node is found and it hasn't been added yet, proceed with the population.
    if (currentNode != null && addedNodes.add(currentNode.id)) {
      // Find all parent edges for the current node.
      final parentEdges =
          _skillTreeEdges.where((edge) => edge.to == currentNode.id);

      // For each parent edge found, recurse to the parent node.
      for (final parentEdge in parentEdges) {
        // Recurse to the parent node identified by the 'from' field of the edge.
        recursivePopulateHierarchy(parentEdge.from, addedNodes);
      }

      // After processing all parent nodes, add the current node to the list.
      _selectedNodeHierarchy!.add(currentNode);
    }
  }

  // Function to get a node by its ID
  SkillTreeNode? getNodeById(String nodeId) {
    try {
      // Find the node in the list where the ID matches
      return _skillTreeNodes.firstWhere((node) => node.id == nodeId);
    } catch (e) {
      print("Node with ID $nodeId not found: $e");
      return null;
    }
  }

  // TODO: Move to task queue view model
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
                task.id, BenchmarkStepRequestBody(input: null));
        Step step = Step.fromMap(stepResponse);

        // Check if it's the last step
        while (!step.isLast) {
          // Fetch chats for the task
          chatViewModel.fetchChatsForTask();

          // Execute next step and update the Step object
          stepResponse = await benchmarkService.executeBenchmarkStep(
              task.id, BenchmarkStepRequestBody(input: null));
          step = Step.fromMap(stepResponse);
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
        // await Future.delayed(Duration(seconds: 2));
        notifyListeners();

        // If successStatus is false, break out of the loop
        if (!successStatus) {
          print(
              "Benchmark for node ${node.id} failed. Stopping all benchmarks.");
          break;
        }
        testSuite.tests.add(task);
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

  // TODO: Move to task queue view model
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
