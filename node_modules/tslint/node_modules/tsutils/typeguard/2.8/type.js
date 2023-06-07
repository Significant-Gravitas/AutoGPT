"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
var ts = require("typescript");
function isConditionalType(type) {
    return (type.flags & ts.TypeFlags.Conditional) !== 0;
}
exports.isConditionalType = isConditionalType;
function isEnumType(type) {
    return (type.flags & ts.TypeFlags.Enum) !== 0;
}
exports.isEnumType = isEnumType;
function isGenericType(type) {
    return (type.flags & ts.TypeFlags.Object) !== 0 &&
        (type.objectFlags & ts.ObjectFlags.ClassOrInterface) !== 0 &&
        (type.objectFlags & ts.ObjectFlags.Reference) !== 0;
}
exports.isGenericType = isGenericType;
function isIndexedAccessType(type) {
    return (type.flags & ts.TypeFlags.IndexedAccess) !== 0;
}
exports.isIndexedAccessType = isIndexedAccessType;
function isIndexedAccessype(type) {
    return (type.flags & ts.TypeFlags.Index) !== 0;
}
exports.isIndexedAccessype = isIndexedAccessype;
function isInstantiableType(type) {
    return (type.flags & ts.TypeFlags.Instantiable) !== 0;
}
exports.isInstantiableType = isInstantiableType;
function isInterfaceType(type) {
    return (type.flags & ts.TypeFlags.Object) !== 0 &&
        (type.objectFlags & ts.ObjectFlags.ClassOrInterface) !== 0;
}
exports.isInterfaceType = isInterfaceType;
function isIntersectionType(type) {
    return (type.flags & ts.TypeFlags.Intersection) !== 0;
}
exports.isIntersectionType = isIntersectionType;
function isLiteralType(type) {
    return (type.flags & ts.TypeFlags.StringOrNumberLiteral) !== 0;
}
exports.isLiteralType = isLiteralType;
function isObjectType(type) {
    return (type.flags & ts.TypeFlags.Object) !== 0;
}
exports.isObjectType = isObjectType;
function isSubstitutionType(type) {
    return (type.flags & ts.TypeFlags.Substitution) !== 0;
}
exports.isSubstitutionType = isSubstitutionType;
function isTypeParameter(type) {
    return (type.flags & ts.TypeFlags.TypeParameter) !== 0;
}
exports.isTypeParameter = isTypeParameter;
function isTypeReference(type) {
    return (type.flags & ts.TypeFlags.Object) !== 0 &&
        (type.objectFlags & ts.ObjectFlags.Reference) !== 0;
}
exports.isTypeReference = isTypeReference;
function isTypeVariable(type) {
    return (type.flags & ts.TypeFlags.TypeVariable) !== 0;
}
exports.isTypeVariable = isTypeVariable;
function isUnionOrIntersectionType(type) {
    return (type.flags & ts.TypeFlags.UnionOrIntersection) !== 0;
}
exports.isUnionOrIntersectionType = isUnionOrIntersectionType;
function isUnionType(type) {
    return (type.flags & ts.TypeFlags.Union) !== 0;
}
exports.isUnionType = isUnionType;
function isUniqueESSymbolType(type) {
    return (type.flags & ts.TypeFlags.UniqueESSymbol) !== 0;
}
exports.isUniqueESSymbolType = isUniqueESSymbolType;
//# sourceMappingURL=data:application/json;base64,eyJ2ZXJzaW9uIjozLCJmaWxlIjoidHlwZS5qcyIsInNvdXJjZVJvb3QiOiIiLCJzb3VyY2VzIjpbInR5cGUudHMiXSwibmFtZXMiOltdLCJtYXBwaW5ncyI6Ijs7QUFBQSwrQkFBaUM7QUFFakMsU0FBZ0IsaUJBQWlCLENBQUMsSUFBYTtJQUMzQyxPQUFPLENBQUMsSUFBSSxDQUFDLEtBQUssR0FBRyxFQUFFLENBQUMsU0FBUyxDQUFDLFdBQVcsQ0FBQyxLQUFLLENBQUMsQ0FBQztBQUN6RCxDQUFDO0FBRkQsOENBRUM7QUFFRCxTQUFnQixVQUFVLENBQUMsSUFBYTtJQUNwQyxPQUFPLENBQUMsSUFBSSxDQUFDLEtBQUssR0FBRyxFQUFFLENBQUMsU0FBUyxDQUFDLElBQUksQ0FBQyxLQUFLLENBQUMsQ0FBQztBQUNsRCxDQUFDO0FBRkQsZ0NBRUM7QUFFRCxTQUFnQixhQUFhLENBQUMsSUFBYTtJQUN2QyxPQUFPLENBQUMsSUFBSSxDQUFDLEtBQUssR0FBRyxFQUFFLENBQUMsU0FBUyxDQUFDLE1BQU0sQ0FBQyxLQUFLLENBQUM7UUFDM0MsQ0FBaUIsSUFBSyxDQUFDLFdBQVcsR0FBRyxFQUFFLENBQUMsV0FBVyxDQUFDLGdCQUFnQixDQUFDLEtBQUssQ0FBQztRQUMzRSxDQUFpQixJQUFLLENBQUMsV0FBVyxHQUFHLEVBQUUsQ0FBQyxXQUFXLENBQUMsU0FBUyxDQUFDLEtBQUssQ0FBQyxDQUFDO0FBQzdFLENBQUM7QUFKRCxzQ0FJQztBQUVELFNBQWdCLG1CQUFtQixDQUFDLElBQWE7SUFDN0MsT0FBTyxDQUFDLElBQUksQ0FBQyxLQUFLLEdBQUcsRUFBRSxDQUFDLFNBQVMsQ0FBQyxhQUFhLENBQUMsS0FBSyxDQUFDLENBQUM7QUFDM0QsQ0FBQztBQUZELGtEQUVDO0FBRUQsU0FBZ0Isa0JBQWtCLENBQUMsSUFBYTtJQUM1QyxPQUFPLENBQUMsSUFBSSxDQUFDLEtBQUssR0FBRyxFQUFFLENBQUMsU0FBUyxDQUFDLEtBQUssQ0FBQyxLQUFLLENBQUMsQ0FBQztBQUNuRCxDQUFDO0FBRkQsZ0RB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