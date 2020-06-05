
import ast


class GATransformer(ast.NodeTransformer):
    """
    This is an AST transformer that converts operations into
    JITable counterparts that work on MultiVector value arrays.
    We crawl the AST and convert BinOp's into numba overloaded
    functions.
    """
    def visit_BinOp(self, node):
        ops = {
            ast.Mult: 'ga_mul',
            ast.BitXor: 'ga_xor',
            ast.BitOr: 'ga_or',
            ast.Add: 'ga_add',
            ast.Sub: 'ga_sub',
        }
        try:
            func_name = ops[type(node.op)]
        except KeyError:
            return node
        else:
            return ast.Call(
                func=ast.Name(id=func_name, ctx=ast.Load()),
                args=[self.visit(node.left), self.visit(node.right)],
                keywords=[]
            )