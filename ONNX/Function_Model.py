import numpy
from onnx import numpy_helper,TensorProto
from onnx.helper import(
    make_model,make_node,set_model_props,make_tensor,make_graph,
    make_tensor_value_info,make_opsetid,
    make_function
)
from onnx.checker import check_model

new_domain = 'custom' #自定义计算域
#使用onnx14版本，自定义算子为1版本
opset_imports = [make_opsetid("",14),make_opsetid(new_domain,1)]

node1 = make_node('MatMul',['X','A'],['XA'])
node2 = make_node('Add',['XA','B'],['Y'])

linear_regression = make_function(
    new_domain,
    'LinearRegression',
    ['X','A','B'],
    ['Y'],
    [node1,node2],
    opset_imports,
    [])

X = make_tensor_value_info('X',TensorProto.FLOAT,[None,None])
A = make_tensor_value_info('A',TensorProto.FLOAT,[None,None])
B = make_tensor_value_info('B',TensorProto.FLOAT,[None,None])
Y = make_tensor_value_info('Y',TensorProto.FLOAT,[None])

graph = make_graph(
    [make_node('LinearRegression',['X','A','B'],['Y1'],domain=new_domain),
     make_node('Abs',['Y1'],['Y'])],
     'example',
     [X,A,B],[Y])

onnx_model = make_model(
    graph,opset_imports=opset_imports,
    functions=[linear_regression])
check_model(onnx_model)
print(onnx_model)
