import 'package:auto_gpt_flutter_client/models/skill_tree/skill_tree_node.dart';
import 'package:auto_gpt_flutter_client/viewmodels/skill_tree_viewmodel.dart';
import 'package:auto_gpt_flutter_client/viewmodels/task_queue_viewmodel.dart';
import 'package:flutter/material.dart';
import 'package:provider/provider.dart';

class TreeNodeView extends StatefulWidget {
  final SkillTreeNode node;
  final bool selected;

  TreeNodeView({required this.node, this.selected = false});

  @override
  _TreeNodeViewState createState() => _TreeNodeViewState();
}

class _TreeNodeViewState extends State<TreeNodeView> {
  bool _isHovering = false;

  @override
  Widget build(BuildContext context) {
    return GestureDetector(
      onTap: () {
        print('Node ${widget.node.id} clicked');
        final taskQueueViewModel =
            Provider.of<TaskQueueViewModel>(context, listen: false);
        if (!taskQueueViewModel.isBenchmarkRunning) {
          final skillTreeViewModel =
              Provider.of<SkillTreeViewModel>(context, listen: false);
          skillTreeViewModel.toggleNodeSelection(widget.node.id);
          taskQueueViewModel.updateSelectedNodeHierarchyBasedOnOption(
              taskQueueViewModel.selectedOption,
              skillTreeViewModel.selectedNode,
              skillTreeViewModel.skillTreeNodes,
              skillTreeViewModel.skillTreeEdges);
        }
      },
      child: MouseRegion(
        onEnter: (_) => setState(() => _isHovering = true),
        onExit: (_) => setState(() => _isHovering = false),
        child: Column(
          mainAxisSize: MainAxisSize.min, // Use minimum space
          children: [
            Container(
              width: 30,
              height: 30,
              decoration: BoxDecoration(
                color: Colors.grey[300], // Light grey
                borderRadius:
                    BorderRadius.circular(8), // Slight rounded corners
              ),
              child: Center(
                child: Icon(
                  Icons.star,
                  color: widget.selected
                      ? Colors.red
                      : (_isHovering
                          ? Colors.red
                          : Colors
                              .black), // Black when not hovering or selected
                ),
              ),
            ),
            SizedBox(height: 4),
            Text(
              widget.node.label,
              style: TextStyle(fontSize: 12),
            ),
          ],
        ),
      ),
    );
  }
}
