import 'package:auto_gpt_flutter_client/models/skill_tree/skill_node_data.dart';

// TODO: Update this with actual data
class SkillTreeNode {
  final String color;
  final int id;

  // final SkillNodeData data;

  SkillTreeNode({required this.color, required this.id});

  factory SkillTreeNode.fromJson(Map<String, dynamic> json) {
    return SkillTreeNode(
      color: json['color'],
      id: json['id'],
    );
  }
}
