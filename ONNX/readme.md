## 创建ONNX 自定义函数  
1.定义计算域  
2.定义算子节点 
from onnx.helper    import make_node  

node1 = make_node("MatMul", ["X", "A"], ["XA"])  
node2 = make_node("Add", ["XA", "B"], ["Y"])    
3.创建自定义函数   
linear_regression = make_function(  
    new_domain,             # domain name  
    'LinearRegression',     # function name  
    ['X', 'A', 'B'],        # input names  
    ['Y'],                  # output names  
    [node1, node2],         # nodes  
    opset_imports,          # opsets  
    [])                     # attribute names  
4.构建计算图  
 
X = make_tensor_value_info('X', TensorProto.FLOAT, [None, None])  
A = make_tensor_value_info('A', TensorProto.FLOAT, [None, None])  
B = make_tensor_value_info('B', TensorProto.FLOAT, [None, None])  
Y = make_tensor_value_info('Y', TensorProto.FLOAT, [None])  

graph = make_graph(
    [make_node('LinearRegression', ['X', 'A', 'B'], ['Y1'],   domain=new_domain),  
     make_node('Abs', ['Y1'], ['Y'])],  
    'example',  
    [X, A, B], [Y])  

5.生成onnx模型  
onnx_model = make_model(  
    graph, opset_imports=opset_imports,  
    functions=[linear_regression]  # functions to add)  
check_model(onnx_model)  

