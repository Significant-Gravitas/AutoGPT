import 'dart:convert';
import 'package:auto_gpt_flutter_client/models/benchmark/benchmark_step_request_body.dart';
import 'package:auto_gpt_flutter_client/models/benchmark/benchmark_task_request_body.dart';
import 'package:auto_gpt_flutter_client/models/skill_tree/skill_tree_edge.dart';
import 'package:auto_gpt_flutter_client/models/skill_tree/skill_tree_node.dart';
import 'package:auto_gpt_flutter_client/models/step.dart';
import 'package:auto_gpt_flutter_client/models/task.dart';
import 'package:auto_gpt_flutter_client/models/test_suite.dart';
import 'package:auto_gpt_flutter_client/services/benchmark_service.dart';
import 'package:auto_gpt_flutter_client/viewmodels/chat_viewmodel.dart';
import 'package:auto_gpt_flutter_client/viewmodels/task_viewmodel.dart';
import 'package:collection/collection.dart';
import 'package:flutter/foundation.dart';
import 'package:flutter/services.dart';
import 'package:graphview/GraphView.dart';

class SkillTreeViewModel extends ChangeNotifier {
  // TODO: Potentially move to task queue view model when we create one
  final BenchmarkService benchmarkService;
  // TODO: Potentially move to task queue view model when we create one
  bool isBenchmarkRunning = false;
  // TODO: Potentially move to task queue view model when we create one
  List<Map<SkillTreeNode, bool>> benchmarkStatusList = [];

  List<SkillTreeNode> _skillTreeNodes = [];
  List<SkillTreeEdge> _skillTreeEdges = [];
  SkillTreeNode? _selectedNode;
  // TODO: Potentially move to task queue view model when we create one
  List<SkillTreeNode>? _selectedNodeHierarchy;

  List<SkillTreeNode> get skillTreeNodes => _skillTreeNodes;
  List<SkillTreeEdge> get skillTreeEdges => _skillTreeEdges;
  SkillTreeNode? get selectedNode => _selectedNode;
  List<SkillTreeNode>? get selectedNodeHierarchy => _selectedNodeHierarchy;

  final Graph graph = Graph()..isTree = true;
  BuchheimWalkerConfiguration builder = BuchheimWalkerConfiguration();

  SkillTreeViewModel(this.benchmarkService);

  Future<void> initializeSkillTree() async {
    try {
      resetState();

      // Read the JSON file from assets
      String jsonContent =
          await rootBundle.loadString('assets/tree_structure.json');

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

      builder
        ..siblingSeparation = (50)
        ..levelSeparation = (50)
        ..subtreeSeparation = (50)
        ..orientation = (BuchheimWalkerConfiguration.ORIENTATION_LEFT_RIGHT);

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

  void populateSelectedNodeHierarchy(String startNodeId) {
    // Initialize an empty list to hold the nodes in the hierarchy.
    _selectedNodeHierarchy = [];

    // Find the starting node (the selected node) in the skill tree nodes list.
    SkillTreeNode? currentNode =
        _skillTreeNodes.firstWhere((node) => node.id == startNodeId);

    // Loop through the tree to populate the hierarchy list.
    // The loop will continue as long as there's a valid current node.
    while (currentNode != null) {
      // Add the current node to the hierarchy list.
      _selectedNodeHierarchy!.add(currentNode);

      // Find the parent node by looking through the skill tree edges.
      // We find the edge where the 'to' field matches the ID of the current node.
      SkillTreeEdge? parentEdge = _skillTreeEdges
          .firstWhereOrNull((edge) => edge.to == currentNode?.id);

      // If a parent edge is found, find the corresponding parent node.
      if (parentEdge != null) {
        // The 'from' field of the edge gives us the ID of the parent node.
        // We find that node in the skill tree nodes list.
        currentNode = _skillTreeNodes
            .firstWhereOrNull((node) => node.id == parentEdge.from);
      } else {
        // If no parent edge is found, it means we've reached the root node.
        // We set currentNode to null to exit the loop.
        currentNode = null;
      }
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
    benchmarkStatusList.clear();

    // Create a new TestSuite object with the current timestamp
    final testSuite =
        TestSuite(timestamp: DateTime.now().toIso8601String(), tests: []);

    // Set the benchmark running flag to true
    isBenchmarkRunning = true;
    // Notify listeners
    notifyListeners();

    // Populate benchmarkStatusList with reversed node hierarchy
    final reversedSelectedNodeHierarchy =
        List.from(_selectedNodeHierarchy!.reversed);
    for (var node in reversedSelectedNodeHierarchy) {
      benchmarkStatusList.add({node: false});
    }

    try {
      // Loop through the nodes in the hierarchy
      for (var node in reversedSelectedNodeHierarchy) {
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
        print("Evaluation response: $evaluationResponse");

        // Update the benchmarkStatusList based on the evaluation response
        bool successStatus = evaluationResponse['metrics']['success'];
        var nodeStatus = benchmarkStatusList.firstWhere(
          (element) => element.keys.first.id == node.id,
        );
        nodeStatus[node] = successStatus;

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
}
