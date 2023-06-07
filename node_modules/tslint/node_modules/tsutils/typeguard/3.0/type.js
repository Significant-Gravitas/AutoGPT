"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
var tslib_1 = require("tslib");
tslib_1.__exportStar(require("../2.9/type"), exports);
var ts = require("typescript");
function isTupleType(type) {
    return (type.flags & ts.TypeFlags.Object && type.objectFlags & ts.ObjectFlags.Tuple) !== 0;
}
exports.isTupleType = isTupleType;
//# sourceMappingURL=data:application/json;base64,eyJ2ZXJzaW9uIjozLCJmaWxlIjoidHlwZS5qcyIsInNvdXJjZVJvb3QiOiIiLCJzb3VyY2VzIjpbInR5cGUudHMiXSwibmFtZXMiOltdLCJtYXBwaW5ncyI6Ijs7O0FBQUEsc0RBQTRCO0FBRTVCLCtCQUFpQztBQUVqQyxTQUFnQixXQUFXLENBQUMsSUFBYTtJQUNyQyxPQUFPLENBQUMsSUFBSSxDQUFDLEtBQUssR0FBRyxFQUFFLENBQUMsU0FBUyxDQUFDLE1BQU0sSUFBb0IsSUFBSyxDQUFDLFdBQVcsR0FBRyxFQUFFLENBQUMsV0FBVyxDQUFDLEtBQUssQ0FBQyxLQUFLLENBQUMsQ0FBQztBQUNoSCxDQUFDO0FBRkQsa0NBRUMifQ==