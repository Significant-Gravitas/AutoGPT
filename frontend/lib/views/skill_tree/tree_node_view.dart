import 'package:auto_gpt_flutter_client/viewmodels/skill_tree_viewmodel.dart';
import 'package:flutter/material.dart';
import 'package:provider/provider.dart';

class TreeNodeView extends StatelessWidget {
  final int nodeId;
  final bool selected;

  TreeNodeView({required this.nodeId, this.selected = false});

  @override
  Widget build(BuildContext context) {
    return InkWell(
      onTap: () {
        print('Node $nodeId clicked');
        Provider.of<SkillTreeViewModel>(context, listen: false)
            .toggleNodeSelection(nodeId);
      },
      child: Container(
        padding: EdgeInsets.all(16),
        decoration: BoxDecoration(
          color: selected ? Colors.red : Colors.white,
          borderRadius: BorderRadius.circular(4),
          boxShadow: [
            BoxShadow(color: Colors.red, spreadRadius: 1),
          ],
        ),
        child: Text('Node $nodeId'),
      ),
    );
  }
}
