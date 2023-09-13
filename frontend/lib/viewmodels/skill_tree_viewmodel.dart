import 'package:auto_gpt_flutter_client/models/skill_tree/skill_tree_edge.dart';
import 'package:auto_gpt_flutter_client/models/skill_tree/skill_tree_node.dart';
import 'package:collection/collection.dart';
import 'package:flutter/foundation.dart';
import 'package:graphview/GraphView.dart';

class SkillTreeViewModel extends ChangeNotifier {
  List<SkillTreeNode> _skillTreeNodes = [];
  List<SkillTreeEdge> _skillTreeEdges = [];
  SkillTreeNode? _selectedNode;
  List<SkillTreeNode>? _selectedNodeHierarchy;

  SkillTreeNode? get selectedNode => _selectedNode;
  List<SkillTreeNode>? get selectedNodeHierarchy => _selectedNodeHierarchy;

  final Graph graph = Graph()..isTree = true;
  BuchheimWalkerConfiguration builder = BuchheimWalkerConfiguration();

  void initializeSkillTree() {
    // TODO: Load from JSON
    resetState();
    // Add nodes to _skillTreeNodes
    _skillTreeNodes.addAll([
      SkillTreeNode(color: 'red', id: 1),
      SkillTreeNode(color: 'blue', id: 2),
      SkillTreeNode(color: 'green', id: 3),
      SkillTreeNode(color: 'yellow', id: 4),
      SkillTreeNode(color: 'orange', id: 5),
      SkillTreeNode(color: 'purple', id: 6),
      SkillTreeNode(color: 'brown', id: 7),
      SkillTreeNode(color: 'pink', id: 8),
      SkillTreeNode(color: 'grey', id: 9),
      SkillTreeNode(color: 'cyan', id: 10),
      SkillTreeNode(color: 'magenta', id: 11),
      SkillTreeNode(color: 'lime', id: 12)
    ]);

    // Add edges to _skillTreeEdges
    _skillTreeEdges.addAll([
      SkillTreeEdge(id: '1_to_2', from: '1', to: '2', arrows: 'to'),
      SkillTreeEdge(id: '1_to_3', from: '1', to: '3', arrows: 'to'),
      SkillTreeEdge(id: '1_to_4', from: '1', to: '4', arrows: 'to'),
      SkillTreeEdge(id: '2_to_5', from: '2', to: '5', arrows: 'to'),
      SkillTreeEdge(id: '2_to_6', from: '2', to: '6', arrows: 'to'),
      SkillTreeEdge(id: '6_to_7', from: '6', to: '7', arrows: 'to'),
      SkillTreeEdge(id: '6_to_8', from: '6', to: '8', arrows: 'to'),
      SkillTreeEdge(id: '4_to_9', from: '4', to: '9', arrows: 'to'),
      SkillTreeEdge(id: '4_to_10', from: '4', to: '10', arrows: 'to'),
      SkillTreeEdge(id: '4_to_11', from: '4', to: '11', arrows: 'to'),
      SkillTreeEdge(id: '11_to_12', from: '11', to: '12', arrows: 'to')
    ]);

    builder
      ..siblingSeparation = (100)
      ..levelSeparation = (150)
      ..subtreeSeparation = (150)
      ..orientation = (BuchheimWalkerConfiguration.ORIENTATION_LEFT_RIGHT);

    notifyListeners();
  }

  void resetState() {
    _skillTreeNodes = [];
    _skillTreeEdges = [];
    _selectedNode = null;
    _selectedNodeHierarchy = null;

    graph.nodes.clear();
    graph.edges.clear();
  }

  void toggleNodeSelection(int nodeId) {
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

  void populateSelectedNodeHierarchy(int startNodeId) {
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
          .firstWhereOrNull((edge) => edge.to == currentNode?.id.toString());

      // If a parent edge is found, find the corresponding parent node.
      if (parentEdge != null) {
        // The 'from' field of the edge gives us the ID of the parent node.
        // We find that node in the skill tree nodes list.
        currentNode = _skillTreeNodes
            .firstWhereOrNull((node) => node.id.toString() == parentEdge.from);
      } else {
        // If no parent edge is found, it means we've reached the root node.
        // We set currentNode to null to exit the loop.
        currentNode = null;
      }
    }
  }

  // Getter to expose nodes for the View
  List<SkillTreeNode> get skillTreeNodes => _skillTreeNodes;

  // Getter to expose edges for the View
  List<SkillTreeEdge> get skillTreeEdges => _skillTreeEdges;
}
