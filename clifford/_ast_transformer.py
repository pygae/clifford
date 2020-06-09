
import ast


class DecoratorRemover(ast.NodeTransformer):
    """ Strip decorators from top-level FunctionDefs"""
    def visit_FunctionDef(self, node):
        node.decorator_list = []
        return node


class GATransformer(ast.NodeTransformer):
    """
    This is an AST transformer that converts operations into
    JITable counterparts that work on MultiVector value arrays.
    We crawl the AST and convert BinOps and UnaryOps into numba
    overloaded functions.
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

    def visit_UnaryOp(self, node):
        ops = {
            ast.Invert: 'ga_rev'
        }
        try:
            func_name = ops[type(node.op)]
        except KeyError:
            return node
        else:
            return ast.Call(
                func=ast.Name(id=func_name, ctx=ast.Load()),
                args=[self.visit(node.operand)],
                keywords=[]
            )

    def visit_Call(self, node):
        if len(node.args) == 1:
            node = self.generic_visit(node)
            node.args = [node.func] + node.args
            node.func = ast.Name(id='ga_call', ctx=ast.Load())
            return node
        else:
            return self.generic_visit(node)
