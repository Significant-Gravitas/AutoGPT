import 'package:auto_gpt_flutter_client/models/skill_tree/skill_tree_category.dart';
import 'package:auto_gpt_flutter_client/models/skill_tree/skill_tree_node.dart';
import 'package:auto_gpt_flutter_client/viewmodels/skill_tree_viewmodel.dart';
import 'package:auto_gpt_flutter_client/views/skill_tree/tree_node_view.dart';
import 'package:flutter/material.dart';
import 'package:graphview/GraphView.dart';

class SkillTreeView extends StatefulWidget {
  final SkillTreeViewModel viewModel;

  const SkillTreeView({Key? key, required this.viewModel}) : super(key: key);

  @override
  _SkillTreeViewState createState() => _SkillTreeViewState();
}

class _SkillTreeViewState extends State<SkillTreeView> {
  Future<void>? initialization;

  @override
  void initState() {
    super.initState();
    initialization = widget.viewModel.initializeSkillTree();
  }

  @override
  Widget build(BuildContext context) {
    return Stack(
      children: [
        FutureBuilder<void>(
          future: initialization,
          builder: (context, snapshot) {
            widget.viewModel.graph.nodes.clear();
            widget.viewModel.graph.edges.clear();
            if (snapshot.connectionState == ConnectionState.waiting) {
              return const CircularProgressIndicator();
            }

            if (snapshot.hasError) {
              return const Text("An error occurred");
            }

            // Create Node and Edge objects for GraphView
            final Map<String, Node> nodeMap = {};
            for (var skillTreeNode in widget.viewModel.skillTreeNodes) {
              final node = Node.Id(skillTreeNode.id);
              widget.viewModel.graph.addNode(node);
              nodeMap[skillTreeNode.id] = node;
            }

            for (var skillTreeEdge in widget.viewModel.skillTreeEdges) {
              final fromNode = nodeMap[skillTreeEdge.from];
              final toNode = nodeMap[skillTreeEdge.to];
              if (fromNode != null && toNode != null) {
                widget.viewModel.graph.addEdge(fromNode, toNode);
              }
            }

            return Scaffold(
              body: Column(
                mainAxisSize: MainAxisSize.max,
                children: [
                  Expanded(
                    child: InteractiveViewer(
                      constrained: false,
                      child: SizedBox(
                        width: MediaQuery.of(context).size.width,
                        height: MediaQuery.of(context).size.height,
                        child: Align(
                          alignment: Alignment.centerLeft,
                          child: GraphView(
                            graph: widget.viewModel.graph,
                            algorithm:
                                SugiyamaAlgorithm(widget.viewModel.builder),
                            paint: Paint()
                              ..color = Colors.green
                              ..strokeWidth = 1
                              ..style = PaintingStyle.stroke,
                            builder: (Node node) {
                              String nodeId = node.key?.value as String;
                              SkillTreeNode? skillTreeNode =
                                  widget.viewModel.getNodeById(nodeId);
                              if (skillTreeNode != null) {
                                return TreeNodeView(
                                    node: skillTreeNode,
                                    selected: nodeId ==
                                        widget.viewModel.selectedNode?.id);
                              } else {
                                return const SizedBox(); // Return an empty widget if the node is not found
                              }
                            },
                          ),
                        ),
                      ),
                    ),
                  ),
                ],
              ),
            );
          },
        ),
        Positioned(
          top: 10,
          left: 10,
          child: Material(
            type: MaterialType.transparency,
            child: DropdownButton<SkillTreeCategory>(
              value: widget.viewModel.currentSkillTreeType,
              items: SkillTreeCategory.values.map((category) {
                return DropdownMenuItem<SkillTreeCategory>(
                  value: category,
                  child: Text(category.stringValue),
                );
              }).toList(),
              onChanged: (newValue) {
                if (newValue != null) {
                  setState(() {
                    widget.viewModel.currentSkillTreeType = newValue;
                    widget.viewModel.initializeSkillTree();
                  });
                }
              },
            ),
          ),
        )
      ],
    );
  }
}
