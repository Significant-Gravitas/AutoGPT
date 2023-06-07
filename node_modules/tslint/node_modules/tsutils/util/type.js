"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
var ts = require("typescript");
var type_1 = require("../typeguard/type");
var util_1 = require("./util");
function isEmptyObjectType(type) {
    if (type_1.isObjectType(type) &&
        type.objectFlags & ts.ObjectFlags.Anonymous &&
        type.getProperties().length === 0 &&
        type.getCallSignatures().length === 0 &&
        type.getConstructSignatures().length === 0 &&
        type.getStringIndexType() === undefined &&
        type.getNumberIndexType() === undefined) {
        var baseTypes = type.getBaseTypes();
        return baseTypes === undefined || baseTypes.every(isEmptyObjectType);
    }
    return false;
}
exports.isEmptyObjectType = isEmptyObjectType;
function removeOptionalityFromType(checker, type) {
    if (!containsTypeWithFlag(type, ts.TypeFlags.Undefined))
        return type;
    var allowsNull = containsTypeWithFlag(type, ts.TypeFlags.Null);
    type = checker.getNonNullableType(type);
    return allowsNull ? checker.getNullableType(type, ts.TypeFlags.Null) : type;
}
exports.removeOptionalityFromType = removeOptionalityFromType;
function containsTypeWithFlag(type, flag) {
    for (var _i = 0, _a = unionTypeParts(type); _i < _a.length; _i++) {
        var t = _a[_i];
        if (util_1.isTypeFlagSet(t, flag))
            return true;
    }
    return false;
}
function isTypeAssignableToNumber(checker, type) {
    return isTypeAssignableTo(checker, type, ts.TypeFlags.NumberLike);
}
exports.isTypeAssignableToNumber = isTypeAssignableToNumber;
function isTypeAssignableToString(checker, type) {
    return isTypeAssignableTo(checker, type, ts.TypeFlags.StringLike);
}
exports.isTypeAssignableToString = isTypeAssignableToString;
function isTypeAssignableTo(checker, type, flags) {
    flags |= ts.TypeFlags.Any;
    var typeParametersSeen;
    return (function check(t) {
        if (type_1.isTypeParameter(t) && t.symbol !== undefined && t.symbol.declarations !== undefined) {
            if (typeParametersSeen === undefined) {
                typeParametersSeen = new Set([t]);
            }
            else if (!typeParametersSeen.has(t)) {
                typeParametersSeen.add(t);
            }
            else {
                return false;
            }
            var declaration = t.symbol.declarations[0];
            if (declaration.constraint === undefined)
                return true;
            return check(checker.getTypeFromTypeNode(declaration.constraint));
        }
        if (type_1.isUnionType(t))
            return t.types.every(check);
        if (type_1.isIntersectionType(t))
            return t.types.some(check);
        return util_1.isTypeFlagSet(t, flags);
    })(type);
}
function getCallSignaturesOfType(type) {
    if (type_1.isUnionType(type)) {
        var signatures = [];
        for (var _i = 0, _a = type.types; _i < _a.length; _i++) {
            var t = _a[_i];
            signatures.push.apply(signatures, getCallSignaturesOfType(t));
        }
        return signatures;
    }
    if (type_1.isIntersectionType(type)) {
        var signatures = void 0;
        for (var _b = 0, _c = type.types; _b < _c.length; _b++) {
            var t = _c[_b];
            var sig = getCallSignaturesOfType(t);
            if (sig.length !== 0) {
                if (signatures !== undefined)
                    return [];
                signatures = sig;
            }
        }
        return signatures === undefined ? [] : signatures;
    }
    return type.getCallSignatures();
}
exports.getCallSignaturesOfType = getCallSignaturesOfType;
function unionTypeParts(type) {
    return type_1.isUnionType(type) ? type.types : [type];
}
exports.unionTypeParts = unionTypeParts;
function isThenableType(checker, node, type) {
    if (type === void 0) { type = checker.getTypeAtLocation(node); }
    for (var _i = 0, _a = unionTypeParts(checker.getApparentType(type)); _i < _a.length; _i++) {
        var ty = _a[_i];
        var then = ty.getProperty('then');
        if (then === undefined)
            continue;
        var thenType = checker.getTypeOfSymbolAtLocation(then, node);
        for (var _b = 0, _c = unionTypeParts(thenType); _b < _c.length; _b++) {
            var t = _c[_b];
            for (var _d = 0, _e = t.getCallSignatures(); _d < _e.length; _d++) {
                var signature = _e[_d];
                if (signature.parameters.length !== 0 && isCallback(checker, signature.parameters[0], node))
                    return true;
            }
        }
    }
    return false;
}
exports.isThenableType = isThenableType;
function isCallback(checker, param, node) {
    var type = checker.getApparentType(checker.getTypeOfSymbolAtLocation(param, node));
    if (param.valueDeclaration.dotDotDotToken) {
        type = type.getNumberIndexType();
        if (type === undefined)
            return false;
    }
    for (var _i = 0, _a = unionTypeParts(type); _i < _a.length; _i++) {
        var t = _a[_i];
        if (t.getCallSignatures().length !== 0)
            return true;
    }
    return false;
}
function isFalsyType(type) {
    if (type.flags & (ts.TypeFlags.Undefined | ts.TypeFlags.Null | ts.TypeFlags.Void))
        return true;
    if (type_1.isLiteralType(type))
        return !type.value;
    if (type.flags & ts.TypeFlags.BooleanLiteral)
        return type.intrinsicName === 'false';
    return false;
}
exports.isFalsyType = isFalsyType;
//# sourceMappingURL=data:application/json;base64,eyJ2ZXJzaW9uIjozLCJmaWxlIjoidHlwZS5qcyIsInNvdXJjZVJvb3QiOiIiLCJzb3VyY2VzIjpbInR5cGUudHMiXSwibmFtZXMiOltdLCJtYXBwaW5ncyI6Ijs7QUFBQSwrQkFBaUM7QUFDakMsMENBQWtIO0FBQ2xILCtCQUF1QztBQUV2QyxTQUFnQixpQkFBaUIsQ0FBQyxJQUFhO0lBQzNDLElBQUksbUJBQVksQ0FBQyxJQUFJLENBQUM7UUFDbEIsSUFBSSxDQUFDLFdBQVcsR0FBRyxFQUFFLENBQUMsV0FBVyxDQUFDLFNBQVM7UUFDM0MsSUFBSSxDQUFDLGFBQWEsRUFBRSxDQUFDLE1BQU0sS0FBSyxDQUFDO1FBQ2pDLElBQUksQ0FBQyxpQkFBaUIsRUFBRSxDQUFDLE1BQU0sS0FBSyxDQUFDO1FBQ3JDLElBQUksQ0FBQyxzQkFBc0IsRUFBRSxDQUFDLE1BQU0sS0FBSyxDQUFDO1FBQzFDLElBQUksQ0FBQyxrQkFBa0IsRUFBRSxLQUFLLFNBQVM7UUFDdkMsSUFBSSxDQUFDLGtCQUFrQixFQUFFLEtBQUssU0FBUyxFQUFFO1FBQ3pDLElBQU0sU0FBUyxHQUFHLElBQUksQ0FBQyxZQUFZLEVBQUUsQ0FBQztRQUN0QyxPQUFPLFNBQVMsS0FBSyxTQUFTLElBQUksU0FBUyxDQUFDLEtBQUssQ0FBQyxpQkFBaUIsQ0FBQyxDQUFDO0tBQ3hFO0lBQ0QsT0FBTyxLQUFLLENBQUM7QUFDakIsQ0FBQztBQVpELDhDQVlDO0FBRUQsU0FBZ0IseUJBQXlCLENBQUMsT0FBdUIsRUFBRSxJQUFhO0lBQzVFLElBQUksQ0FBQyxvQkFBb0IsQ0FBQyxJQUFJLEVBQUUsRUFBRSxDQUFDLFNBQVMsQ0FBQyxTQUFTLENBQUM7UUFDbkQsT0FBTyxJQUFJLENBQUM7SUFDaEIsSUFBTSxVQUFVLEdBQUcsb0JBQW9CLENBQUMsSUFBSSxFQUFFLEVBQUUsQ0FBQyxTQUFTLENBQUMsSUFBSSxDQUFDLENBQUM7SUFDakUsSUFBSSxHQUFHLE9BQU8sQ0FBQyxrQkFBa0IsQ0FBQyxJQUFJLENBQUMsQ0FBQztJQUN4QyxPQUFPLFVB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