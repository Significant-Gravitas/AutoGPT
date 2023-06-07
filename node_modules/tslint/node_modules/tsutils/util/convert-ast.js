"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
var ts = require("typescript");
var util_1 = require("./util");
function convertAst(sourceFile) {
    var wrapped = {
        node: sourceFile,
        parent: undefined,
        kind: ts.SyntaxKind.SourceFile,
        children: [],
        next: undefined,
        skip: undefined,
    };
    var flat = [];
    var current = wrapped;
    var previous = current;
    ts.forEachChild(sourceFile, function wrap(node) {
        flat.push(node);
        var parent = current;
        previous.next = current = {
            node: node,
            parent: parent,
            kind: node.kind,
            children: [],
            next: undefined,
            skip: undefined,
        };
        if (previous !== parent)
            setSkip(previous, current);
        previous = current;
        parent.children.push(current);
        if (util_1.isNodeKind(node.kind))
            ts.forEachChild(node, wrap);
        current = parent;
    });
    return {
        wrapped: wrapped,
        flat: flat,
    };
}
exports.convertAst = convertAst;
function setSkip(node, skip) {
    do {
        node.skip = skip;
        node = node.parent;
    } while (node !== skip.parent);
}
//# sourceMappingURL=data:application/json;base64,eyJ2ZXJzaW9uIjozLCJmaWxlIjoiY29udmVydC1hc3QuanMiLCJzb3VyY2VSb290IjoiIiwic291cmNlcyI6WyJjb252ZXJ0LWFzdC50cyJdLCJuYW1lcyI6W10sIm1hcHBpbmdzIjoiOztBQUFBLCtCQUFpQztBQUNqQywrQkFBb0M7QUFtQ3BDLFNBQWdCLFVBQVUsQ0FBQyxVQUF5QjtJQUNoRCxJQUFNLE9BQU8sR0FBZTtRQUN4QixJQUFJLEVBQUUsVUFBVTtRQUNoQixNQUFNLEVBQUUsU0FBUztRQUNqQixJQUFJLEVBQUUsRUFBRSxDQUFDLFVBQVUsQ0FBQyxVQUFVO1FBQzlCLFFBQVEsRUFBRSxFQUFFO1FBQ1osSUFBSSxFQUFPLFNBQVM7UUFDcEIsSUFBSSxFQUFFLFNBQVM7S0FDbEIsQ0FBQztJQUNGLElBQU0sSUFBSSxHQUFjLEVBQUUsQ0FBQztJQUMzQixJQUFJLE9BQU8sR0FBYSxPQUFPLENBQUM7SUFDaEMsSUFBSSxRQUFRLEdBQUcsT0FBTyxDQUFDO0lBQ3ZCLEVBQUUsQ0FBQyxZQUFZLENBQUMsVUFBVSxFQUFFLFNBQVMsSUFBSSxDQUFDLElBQUk7UUFDMUMsSUFBSSxDQUFDLElBQUksQ0FBQyxJQUFJLENBQUMsQ0FBQztRQUNoQixJQUFNLE1BQU0sR0FBRyxPQUFPLENBQUM7UUFDdkIsUUFBUSxDQUFDLElBQUksR0FBRyxPQUFPLEdBQUc7WUFDdEIsSUFBSSxNQUFBO1lBQ0osTUFBTSxRQUFBO1lBQ04sSUFBSSxFQUFFLElBQUksQ0FBQyxJQUFJO1lBQ2YsUUFBUSxFQUFFLEVBQUU7WUFDWixJQUFJLEVBQUUsU0FBUztZQUNmLElBQUksRUFBRSxTQUFTO1NBQ2xCLENBQUM7UUFDRixJQUFJLFFBQVEsS0FBSyxNQUFNO1lBQ25CLE9BQU8sQ0FBQyxRQUFRLEVBQUUsT0FBTyxDQUFDLENBQUM7UUFFL0IsUUFBUSxHQUFHLE9BQU8sQ0FBQztRQUNuQixNQUFNLENBQUMsUUFBUSxDQUFDLElBQUksQ0FBQyxPQUFPLENBQUMsQ0FBQztRQUU5QixJQUFJLGlCQUFVLENBQUMsSUFBSSxDQUFDLElBQUksQ0FBQztZQUNyQixFQUFFLENBQUMsWUFBWSxDQUFDLElBQUksRUFBRSxJQUFJLENBQUMsQ0FBQztRQUVoQyxPQUFPLEdBQUcsTUFBTSxDQUFDO0lBQ3JCLENBQUMsQ0FBQyxDQUFDO0lBRUgsT0FBTztRQUNILE9BQU8sU0FBQTtRQUNQLElBQUksTUFBQTtLQUNQLENBQUM7QUFDTixDQUFDO0FBdkNELGdDQXVDQztBQUVELFNBQVMsT0FBTyxDQUFDLElBQWMsRUFBRSxJQUFjO0lBQzNDLEdBQUc7UUFDQyxJQUFJLENBQUMsSUFBSSxHQUFHLElBQUksQ0FBQztRQUNqQixJQUFJLEdBQUcsSUFBSSxDQUFDLE1BQU8sQ0FBQztLQUN2QixRQUFRLElBQUksS0FBSyxJQUFJLENBQUMsTUFBTSxFQUFFO0FBQ25DLENBQUMifQ==