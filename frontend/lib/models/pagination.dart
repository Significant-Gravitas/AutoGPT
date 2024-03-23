class Pagination {
  final int totalItems;
  final int totalPages;
  final int currentPage;
  final int pageSize;

  Pagination({
    required this.totalItems,
    required this.totalPages,
    required this.currentPage,
    required this.pageSize,
  });

  factory Pagination.fromJson(Map<String, dynamic> json) {
    return Pagination(
      totalItems: json['total_items'],
      totalPages: json['total_pages'],
      currentPage: json['current_page'],
      pageSize: json['page_size'],
    );
  }
}
