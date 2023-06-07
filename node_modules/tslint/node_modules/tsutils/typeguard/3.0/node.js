"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
var tslib_1 = require("tslib");
tslib_1.__exportStar(require("../2.9/node"), exports);
var ts = require("typescript");
function isOptionalTypeNode(node) {
    return node.kind === ts.SyntaxKind.OptionalType;
}
exports.isOptionalTypeNode = isOptionalTypeNode;
function isRestTypeNode(node) {
    return node.kind === ts.SyntaxKind.RestType;
}
exports.isRestTypeNode = isRestTypeNode;
function isSyntheticExpression(node) {
    return node.kind === ts.SyntaxKind.SyntheticExpression;
}
exports.isSyntheticExpression = isSyntheticExpression;
//# sourceMappingURL=data:application/json;base64,eyJ2ZXJzaW9uIjozLCJmaWxlIjoibm9kZS5qcyIsInNvdXJjZVJvb3QiOiIiLCJzb3VyY2VzIjpbIm5vZGUudHMiXSwibmFtZXMiOltdLCJtYXBwaW5ncyI6Ijs7O0FBQUEsc0RBQTRCO0FBRTVCLCtCQUFpQztBQUVqQyxTQUFnQixrQkFBa0IsQ0FBQyxJQUFhO0lBQzVDLE9BQU8sSUFBSSxDQUFDLElBQUksS0FBSyxFQUFFLENBQUMsVUFBVSxDQUFDLFlBQVksQ0FBQztBQUNwRCxDQUFDO0FBRkQsZ0RBRUM7QUFFRCxTQUFnQixjQUFjLENBQUMsSUFBYTtJQUN4QyxPQUFPLElBQUksQ0FBQyxJQUFJLEtBQUssRUFBRSxDQUFDLFVBQVUsQ0FBQyxRQUFRLENBQUM7QUFDaEQsQ0FBQztBQUZELHdDQUVDO0FBRUQsU0FBZ0IscUJBQXFCLENBQUMsSUFBYTtJQUMvQyxPQUFPLElBQUksQ0FBQyxJQUFJLEtBQUssRUFBRSxDQUFDLFVBQVUsQ0FBQyxtQkFBbUIsQ0FBQztBQUMzRCxDQUFDO0FBRkQsc0RBRUMifQ==