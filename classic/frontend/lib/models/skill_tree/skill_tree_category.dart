enum SkillTreeCategory {
  general,
  coding,
  data,
  scrapeSynthesize,
}

extension SkillTreeTypeExtension on SkillTreeCategory {
  String get stringValue {
    switch (this) {
      case SkillTreeCategory.general:
        return 'General';
      case SkillTreeCategory.coding:
        return 'Coding';
      case SkillTreeCategory.data:
        return 'Data';
      case SkillTreeCategory.scrapeSynthesize:
        return 'Scrape/Synthesize';
      default:
        return '';
    }
  }

  String get jsonFileName {
    switch (this) {
      case SkillTreeCategory.general:
        return 'general_tree_structure.json';
      case SkillTreeCategory.coding:
        return 'coding_tree_structure.json';
      case SkillTreeCategory.data:
        return 'data_tree_structure.json';
      case SkillTreeCategory.scrapeSynthesize:
        return 'scrape_synthesize_tree_structure.json';
      default:
        return '';
    }
  }
}
