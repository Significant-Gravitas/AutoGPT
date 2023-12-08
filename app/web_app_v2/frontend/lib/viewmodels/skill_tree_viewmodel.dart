import 'dart:convert';
import 'package:flutter/foundation.dart';
import 'package:flutter/services.dart';
import 'package:graphview/GraphView.dart';

import 'package:auto_gpt_flutter_client/models/skill_tree/skill_tree_category.dart';
import 'package:auto_gpt_flutter_client/models/skill_tree/skill_tree_edge.dart';
import 'package:auto_gpt_flutter_client/models/skill_tree/skill_tree_node.dart';

class SkillTreeViewModel extends ChangeNotifier {
  List<SkillTreeNode> _skillTreeNodes = [];
  List<SkillTreeNode> get skillTreeNodes => _skillTreeNodes;

  List<SkillTreeEdge> _skillTreeEdges = [];
  List<SkillTreeEdge> get skillTreeEdges => _skillTreeEdges;

  SkillTreeNode? _selectedNode;
  SkillTreeNode? get selectedNode => _selectedNode;

  final Graph graph = Graph();
  SugiyamaConfiguration builder = SugiyamaConfiguration();

  SkillTreeCategory currentSkillTreeType = SkillTreeCategory.general;

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
  }

  void toggleNodeSelection(String nodeId) {
    if (_selectedNode?.id == nodeId) {
      // Unselect the node if it's already selected
      _selectedNode = null;
    } else {
      // Select the new node
      _selectedNode = _skillTreeNodes.firstWhere((node) => node.id == nodeId);
    }
    notifyListeners();
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
}
