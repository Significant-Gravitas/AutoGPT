import 'package:auto_gpt_flutter_client/viewmodels/skill_tree_viewmodel.dart';
import 'package:auto_gpt_flutter_client/views/skill_tree/tree_node_view.dart';
import 'package:flutter/material.dart';
import 'package:graphview/GraphView.dart';

class SkillTreeView extends StatefulWidget {
  final SkillTreeViewModel viewModel;

  const SkillTreeView({Key? key, required this.viewModel}) : super(key: key);

  @override
  _TreeViewPageState createState() => _TreeViewPageState();
}

class _TreeViewPageState extends State<SkillTreeView> {
  @override
  void initState() {
    super.initState();

    widget.viewModel.initializeSkillTree();

    // Create Node and Edge objects for GraphView
    final Map<int, Node> nodeMap = {};
    for (var skillTreeNode in widget.viewModel.skillTreeNodes) {
      final node = Node.Id(skillTreeNode.id);
      widget.viewModel.graph.addNode(node);
      nodeMap[skillTreeNode.id] = node;
    }

    for (var skillTreeEdge in widget.viewModel.skillTreeEdges) {
      final fromNode = nodeMap[int.parse(skillTreeEdge.from)];
      final toNode = nodeMap[int.parse(skillTreeEdge.to)];
      widget.viewModel.graph.addEdge(fromNode!, toNode!);
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: Column(
        mainAxisSize: MainAxisSize.max,
        children: [
          Expanded(
            child: InteractiveViewer(
              constrained: false,
              boundaryMargin: EdgeInsets.all(100),
              minScale: 0.01,
              maxScale: 5.6,
              child: GraphView(
                graph: widget.viewModel.graph,
                algorithm: BuchheimWalkerAlgorithm(widget.viewModel.builder,
                    TreeEdgeRenderer(widget.viewModel.builder)),
                paint: Paint()
                  ..color = Colors.green
                  ..strokeWidth = 1
                  ..style = PaintingStyle.stroke,
                builder: (Node node) {
                  int nodeId = node.key?.value as int;
                  return TreeNodeView(
                      nodeId: nodeId,
                      selected: nodeId == widget.viewModel.selectedNode?.id);
                },
              ),
            ),
          ),
        ],
      ),
    );
  }
}
