import 'package:auto_gpt_flutter_client/models/skill_tree/skill_node_data.dart';

class SkillTreeNode {
  final String color;
  final SkillNodeData data;
  final String id;
  final String label;
  final String shape;

  SkillTreeNode({
    required this.color,
    required this.data,
    required this.id,
    required this.label,
    required this.shape,
  });

  factory SkillTreeNode.fromJson(Map<String, dynamic> json) {
    return SkillTreeNode(
      color: json['color'] ?? "",
      data: SkillNodeData.fromJson(json['data'] ?? {}),
      id: json['id'] ?? "",
      label: json['label'] ?? "",
      shape: json['shape'] ?? "",
    );
  }
}
