
import ast


class GATransformer(ast.NodeTransformer):
    """
    This is an AST transformer that converts operations into
    JITable counterparts that work on MultiVector value arrays.
    We crawl the AST and convert BinOp's into numba overloaded
    functions.
    """
    def visit_BinOp(self, node):
        if isinstance(node.op, ast.Mult):
            new_node = ast.Call(
                func=ast.Name(id='ga_mul', ctx=ast.Load()),
                args=[node.left, node.right],
                keywords=[]
            )
            new_node = GATransformer().visit(new_node)
            return new_node
        elif isinstance(node.op, ast.BitXor):
            new_node = ast.Call(
                func=ast.Name(id='ga_xor', ctx=ast.Load()),
                args=[node.left, node.right],
                keywords=[]
            )
            new_node = GATransformer().visit(new_node)
            return new_node
        elif isinstance(node.op, ast.BitOr):
            new_node = ast.Call(
                func=ast.Name(id='ga_or', ctx=ast.Load()),
                args=[node.left, node.right],
                keywords=[]
            )
            new_node = GATransformer().visit(new_node)
            return new_node
        elif isinstance(node.op, ast.Add):
            new_node = ast.Call(
                func=ast.Name(id='ga_add', ctx=ast.Load()),
                args=[node.left, node.right],
                keywords=[]
            )
            new_node = GATransformer().visit(new_node)
            return new_node
        elif isinstance(node.op, ast.Sub):
            new_node = ast.Call(
                func=ast.Name(id='ga_sub', ctx=ast.Load()),
                args=[node.left, node.right],
                keywords=[]
            )
            new_node = GATransformer().visit(new_node)
            return new_node
        return node

