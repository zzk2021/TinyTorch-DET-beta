import torch
import torch.nn as nn

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10, 20)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(20, 1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = x + x
        #x2 = x[:,:10]
        return self.fc2(x)


model = Net()
dummy_input = torch.randn(1, 10)

# 获取追踪的计算图
traced = torch.jit.trace(model, dummy_input)
graph = traced.graph

# 打印计算图结构
print(graph)

torch.onnx.export(
    model,
    dummy_input,
    "model.onnx",
    opset_version=13,
    input_names=['input'],
    output_names=['output'],
)

import onnx

# 加载 ONNX 模型
model_path = "model.onnx"
model = onnx.load(model_path)

# 方法 1：使用 onnx.helper.printable_graph
print("="*50, "计算图结构", "="*50)
print(onnx.helper.printable_graph(model.graph))

# 方法 2：直接访问图属性
print("\n" + "="*50, "详细节点信息", "="*50)
for i, node in enumerate(model.graph.node):
    print(f"节点 {i+1}: {node.name}")
    print(f"  操作类型: {node.op_type}")
    print(f"  输入: {node.input}")
    print(f"  输出: {node.output}")

    # 打印属性
    if node.attribute:
        print("  属性:")
        for attr in node.attribute:
            print(f"    {attr.name}: {attr}")

    print("-"*60)

# 打印输入输出信息
print("\n" + "="*50, "模型输入输出", "="*50)
print("输入:")
for input in model.graph.input:
    print(f"  {input.name}: {input.type.tensor_type.elem_type}, 形状: {[dim.dim_value for dim in input.type.tensor_type.shape.dim]}")

print("\n输出:")
for output in model.graph.output:
    print(f"  {output.name}: {output.type.tensor_type.elem_type}, 形状: {[dim.dim_value for dim in output.type.tensor_type.shape.dim]}")