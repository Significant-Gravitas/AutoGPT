import 'package:auto_gpt_flutter_client/models/skill_tree/ground.dart';
import 'package:auto_gpt_flutter_client/models/skill_tree/info.dart';

class SkillNodeData {
  final String name;
  final List<String> category;
  final String task;
  final List<String> dependencies;
  final int cutoff;
  final Ground ground;
  final Info info;

  SkillNodeData({
    required this.name,
    required this.category,
    required this.task,
    required this.dependencies,
    required this.cutoff,
    required this.ground,
    required this.info,
  });

  factory SkillNodeData.fromJson(Map<String, dynamic> json) {
    return SkillNodeData(
      name: json['name'],
      category: List<String>.from(json['category']),
      task: json['task'],
      dependencies: List<String>.from(json['dependencies']),
      cutoff: json['cutoff'],
      ground: Ground.fromJson(json['ground']),
      info: Info.fromJson(json['info']),
    );
  }
}
