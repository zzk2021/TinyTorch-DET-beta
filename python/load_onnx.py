import onnx

# 加载模型
model = onnx.load("model.onnx")

# 遍历所有 initializer（即权重）
for initializer in model.graph.initializer:
    print(f"Name: {initializer.name}")
    print(f"Shape: {initializer.dims}")
    print(f"Data Type: {initializer.data_type}")
    print(f"Raw Data Size (bytes): {len(initializer.raw_data)}")

def parse_onnx_model(model_path):
    model = onnx.load(model_path)
    graph = model.graph

    # 解析输入输出
    inputs = [i.name for i in graph.input]
    outputs = [o.name for o in graph.output]

    # 解析初始值（权重）
    initializers = {init.name: onnx.numpy_helper.to_array(init)
                    for init in graph.initializer}

    # 解析计算节点
    nodes = []
    for node in graph.node:
        node_info = {
            'name': node.name,
            'op_type': node.op_type,
            'inputs': node.input,
            'outputs': node.output,
            'attributes': {attr.name: attr for attr in node.attribute}
        }
        nodes.append(node_info)

    return {
        'inputs': inputs,
        'outputs': outputs,
        'initializers': initializers,
        'nodes': nodes
    }


from collections import deque

def topological_sort(nodes):
    # 构建依赖图
    graph = {node['name']: [] for node in nodes}
    in_degree = {node['name']: 0 for node in nodes}
    name_to_node = {node['name']: node for node in nodes}

    # 建立连接关系
    for node in nodes:
        for output in node['outputs']:
            for next_node in nodes:
                if output in next_node['inputs']:
                    graph[node['name']].append(next_node['name'])
                    in_degree[next_node['name']] += 1

    # 拓扑排序
    queue = deque([name for name, deg in in_degree.items() if deg == 0])
    sorted_nodes = []

    while queue:
        name = queue.popleft()
        sorted_nodes.append(name_to_node[name])
        for neighbor in graph[name]:
            in_degree[neighbor] -= 1
            if in_degree[neighbor] == 0:
                queue.append(neighbor)

    return sorted_nodes

def infer_shapes(model):
    # 使用ONNX内置的形状推断
    from onnx import shape_inference
    inferred_model = shape_inference.infer_shapes(model)

    shape_map = {}
    for value_info in inferred_model.graph.value_info:
        shape = [dim.dim_value for dim in value_info.type.tensor_type.shape.dim]
        shape_map[value_info.name] = shape

    # 添加输入输出形状
    for input in inferred_model.graph.input:
        shape = [dim.dim_value for dim in input.type.tensor_type.shape.dim]
        shape_map[input.name] = shape

    for output in inferred_model.graph.output:
        shape = [dim.dim_value for dim in output.type.tensor_type.shape.dim]
        shape_map[output.name] = shape

    return shape_map


def generate_conv_code(inputs, attrs):
    # 解析属性
    strides = attrs.get('strides', [1, 1])
    pads = attrs.get('pads', [0, 0, 0, 0])
    dilations = attrs.get('dilations', [1, 1])
    group = attrs.get('group', 1)

    return f"conv2d({inputs[0]}, {inputs[1]}, stride={strides}, padding={pads}, dilation={dilations}, groups={group})"

def generate_pool_code(inputs, attrs):
    # 解析属性
    strides = attrs.get('strides', [1, 1])
    pads = attrs.get('pads', [0, 0, 0, 0])
    dilations = attrs.get('dilations', [1, 1])
    group = attrs.get('group', 1)

    return f"pool({inputs[0]}, {inputs[1]}, stride={strides}, padding={pads}, dilation={dilations}, groups={group})"

def generate_reshape_code(inputs, attrs):
    # 解析属性
    strides = attrs.get('strides', [1, 1])
    pads = attrs.get('pads', [0, 0, 0, 0])
    dilations = attrs.get('dilations', [1, 1])
    group = attrs.get('group', 1)

    return f"pool({inputs[0]}, {inputs[1]}, stride={strides}, padding={pads}, dilation={dilations}, groups={group})"
def generate_transpose_code(inputs, attrs):
    # 解析属性
    strides = attrs.get('strides', [1, 1])
    pads = attrs.get('pads', [0, 0, 0, 0])
    dilations = attrs.get('dilations', [1, 1])
    group = attrs.get('group', 1)

    return f"pool({inputs[0]}, {inputs[1]}, stride={strides}, padding={pads}, dilation={dilations}, groups={group})"
def generate_concat_code(inputs, attrs):
    # 解析属性
    strides = attrs.get('strides', [1, 1])
    pads = attrs.get('pads', [0, 0, 0, 0])
    dilations = attrs.get('dilations', [1, 1])
    group = attrs.get('group', 1)

    return f"pool({inputs[0]}, {inputs[1]}, stride={strides}, padding={pads}, dilation={dilations}, groups={group})"
def generate_batchnorm_code(inputs, attrs):
    # 解析属性
    strides = attrs.get('strides', [1, 1])
    pads = attrs.get('pads', [0, 0, 0, 0])
    dilations = attrs.get('dilations', [1, 1])
    group = attrs.get('group', 1)

    return f"pool({inputs[0]}, {inputs[1]}, stride={strides}, padding={pads}, dilation={dilations}, groups={group})"

OP_MAPPING = {
    # 基础操作
    'Add': lambda inputs, attrs: f"{inputs[0]} + {inputs[1]}",
    'Mul': lambda inputs, attrs: f"{inputs[0]} * {inputs[1]}",
    'Sub': lambda inputs, attrs: f"{inputs[0]} - {inputs[1]}",
    'Div': lambda inputs, attrs: f"{inputs[0]} / {inputs[1]}",
    'Pow': lambda inputs, attrs: f"pow({inputs[0]}, {inputs[1]})",
    'Sqrt': lambda inputs, attrs: f"sqrt({inputs[0]})",

    # 激活函数
    'Relu': lambda inputs, attrs: f"Function::relu({inputs[0]})",
    'Sigmoid': lambda inputs, attrs: f"Function::sigmoid({inputs[0]})",
    'Tanh': lambda inputs, attrs: f"Function::tanh({inputs[0]})",
    'LogSoftmax': lambda inputs, attrs: f"Function::logSoftmax({inputs[0]}, {attrs.get('axis', 1)})",

    # 卷积操作
    'Conv': lambda inputs, attrs: generate_conv_code(inputs, attrs),

    # 池化
    'MaxPool': lambda inputs, attrs: f"Function::maxPool2d({inputs[0]}, {attrs['kernel_shape']})",
    'AveragePool': lambda inputs, attrs: f"Function::avgPool2d({inputs[0]}, {attrs['kernel_shape']})",

    # 全连接
    'Gemm': lambda inputs, attrs: generate_gemm_code(inputs, attrs),

    # 张量操作
    'Reshape': lambda inputs, attrs: f"Tensor::reshape({inputs[0]}, {inputs[1]})",
    'Flatten': lambda inputs, attrs: f"Tensor::flatten({inputs[0]}, {attrs.get('axis', 1)})",
    'Concat': lambda inputs, attrs: f"Tensor::concat({{{', '.join(inputs)}}}, {attrs.get('axis', 0)})",
    'Transpose': lambda inputs, attrs: f"Tensor::transpose({inputs[0]}, {attrs['perm']})",

    # 归一化和正则化
    'BatchNormalization': lambda inputs, attrs: generate_batchnorm_code(inputs, attrs),
    'Dropout': lambda inputs, attrs: f"Function::dropout({inputs[0]}, {attrs.get('ratio', 0.5)})",
}


class LayerManager:
    def __init__(self):
        self.layer_counter = {
            'Conv': 0,
            'Linear': 0,
            'Dropout': 0,
            'BatchNorm': 0
        }
        self.layer_map = {}  # 存储节点到层名称的映射

    def get_layer_name(self, node):
        op_type = node['op_type']
        if op_type in ['Conv']:
            self.layer_counter['Conv'] += 1
            return f"conv{self.layer_counter['Conv']}"
        elif op_type in ['Gemm']:
            self.layer_counter['Linear'] += 1
            return f"fc{self.layer_counter['Linear']}"
        elif op_type in ['Dropout']:
            self.layer_counter['Dropout'] += 1
            return f"dropout{self.layer_counter['Dropout']}"
        elif op_type in ['BatchNormalization']:
            self.layer_counter['BatchNorm'] += 1
            return f"batchnorm{self.layer_counter['BatchNorm']}"
        return None

    def add_layer(self, node, layer_name):
        self.layer_map[node['name']] = (node['op_type'], layer_name)

    def get_layer_for_node(self, node_name):
        return self.layer_map.get(node_name)


def generate_conv_code(inputs, attrs):
    # 实际实现中需要解析属性
    return f"{inputs[0]}"  # 实际调用会被替换为成员函数


def generate_gemm_code(inputs, attrs):
    # 实际实现中需要解析属性
    return f"{inputs[0]}"  # 实际调用会被替换为成员函数


def generate_batchnorm_code(inputs, attrs):
    # 实际实现中需要解析属性
    return f"{inputs[0]}"  # 实际调用会被替换为成员函数


class VariableManager:
    def __init__(self):
        self.var_map = {}  # ONNX名称 -> 框架变量名
        self.counter = 0

    def get_or_create_var(self, onnx_name):
        if onnx_name not in self.var_map:
            self.counter += 1
            self.var_map[onnx_name] = f"x{self.counter}"
        return self.var_map[onnx_name]

    def get_var(self, onnx_name):
        return self.var_map.get(onnx_name, f"<unknown:{onnx_name}>")


def generate_framework_code(onnx_model_path, output_file):
    # 解析ONNX模型（伪实现）
    model_info = parse_onnx_model(onnx_model_path)
    shape_map = infer_shapes(onnx.load(onnx_model_path))

    # 拓扑排序节点
    sorted_nodes = topological_sort(model_info['nodes'])

    # 初始化管理器
    var_manager = VariableManager()
    layer_manager = LayerManager()

    # 创建代码缓冲区
    code_lines = [
        "// Auto-generated code from ONNX model",
        "#include \"torch.h\"",
        "",
        "class Net : public nn::Module {",
        "public:"
    ]

    # 1. 生成构造函数和成员注册
    constructor_lines = [
        "    Net() {"
    ]

    # 2. 前向传播函数
    forward_lines = [
        "    ",
        "    Tensor forward(Tensor &x) override {"
    ]

    # 3. 私有成员声明
    private_lines = [
        "private:"
    ]

    # 处理输入
    input_var = var_manager.get_or_create_var(model_info['inputs'][0])
    forward_lines.append(f"        auto {input_var} = x;")

    # 第一遍：识别所有需要作为成员变量的层
    layer_declarations = {}
    layer_registrations = {}
    for node in sorted_nodes:
        layer_name = layer_manager.get_layer_name(node)
        if layer_name:
            layer_manager.add_layer(node, layer_name)
            layer_declarations[node['outputs'][0]] = (node['op_type'], layer_name)
            # 生成成员注册
            if node['op_type'] == 'Conv':
                # 实际参数应从属性中解析
                layer_registrations[layer_name] = f"        {layer_name} = registerModule(\"{layer_name}\", nn::Conv2D(1, 32, 3, 1));"
                private_lines.append(f"    nn::Conv2D {layer_name};")
            elif node['op_type'] == 'Gemm':
                # 实际参数应从属性中解析
                layer_registrations[layer_name] = f"        {layer_name} = registerModule(\"{layer_name}\", nn::Linear(128, 10));"
                private_lines.append(f"    nn::Linear {layer_name};")
            elif node['op_type'] == 'Dropout':
                ratio = node.get('attributes', {}).get('ratio', 0.5)
                layer_registrations[layer_name] = f"        {layer_name} = registerModule(\"{layer_name}\", nn::Dropout({ratio}));"
                private_lines.append(f"    nn::Dropout {layer_name};")
            elif node['op_type'] == 'BatchNormalization':
                # 实际参数应从属性中解析
                layer_registrations[layer_name] = f"        {layer_name} = registerModule(\"{layer_name}\", nn::BatchNorm2D(64));"
                private_lines.append(f"    nn::BatchNorm2D {layer_name};")

    # 添加注册代码到构造函数
    for reg in layer_registrations.values():
        constructor_lines.append(reg)
    constructor_lines.append("    }")
    i = 0
    # 第二遍：生成前向传播代码
    for node in sorted_nodes:
        inputs = [var_manager.get_or_create_var(i) for i in node['inputs']]
        outputs = [var_manager.get_or_create_var(o) for o in node['outputs']]
        output_var = outputs[0]
        print("    ", i,"\n")
        print(node)
        i+=1
        # 检查是否为层操作
        layer_info = layer_manager.get_layer_for_node(node['name'])
        print("layer_info", layer_info)
        if layer_info:
            op_type, layer_name = layer_info
            forward_lines.append(f"        {output_var} = {layer_name}({inputs[0]});")
        elif node['op_type'] in OP_MAPPING:
                expr = OP_MAPPING[node['op_type']](inputs, node.get('attributes', {}))
                forward_lines.append(f"        {output_var} = {expr};")
        else:
            forward_lines.append(f"        // UNSUPPORTED OP: {node['op_type']}")
            forward_lines.append(f"        Tensor {output_var}; // Placeholder for unsupported op")

    # 添加返回语句
    output_var = var_manager.get_or_create_var(model_info['outputs'][0])
    forward_lines.append(f"        return {output_var};")
    forward_lines.append("    }")

    # 合并所有代码
    final_code = []
    final_code.extend(code_lines)           # 头部和类定义
    final_code.extend(constructor_lines)     # 构造函数
    final_code.extend(forward_lines)         # 前向传播函数
    final_code.extend(private_lines)         # 私有成员声明
    final_code.append("};")                  # 类结束

    # 写入文件
    with open(output_file, 'w') as f:
        f.write("\n".join(final_code))

    print(f"Generated code saved to {output_file}")

# 使用示例
generate_framework_code("model.onnx", "p.cpp")