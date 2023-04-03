import os
from cuda import cudart
import cv2 as cv
import numpy as np
import tensorrt as trt
import time
from models import model_build
from torchvision.datasets import MNIST
from utils import Mydataset
from torch.utils.data import DataLoader


def run():
    bs = 8
    n_w = 4
    bUseModel = None
    bUseModelList = [None, 'FP16', 'INT8']
    nHeight, nWidth = 28, 28
    paraFile = './para.npz'
    onnxFile = './model.onnx'

    #为network中添加计算图的具体内容，可以通过1.tensorRT原生API搭建； 2.导入onnx文件进行Parser
    useSourceApi = True
    start = time.time()
    precision = 4
    # 记录日志文件
    logger = trt.Logger(trt.Logger.ERROR)

    # 作为模型搭建的入口，用于产生推理的engine和对网络进行配置、优化
    builder = trt.Builder(logger)

    # 构建网络的主体，可以通过三个方法来构建；对网络设置为EXPLICIT_BATCH的表示（模型中batch设置为显性）
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))

    # 产生用于网络优化
    profile = builder.create_optimization_profile()

    # 进行网络的配置
    config = builder.create_builder_config()
        # 1.设置模型推理的精度； 
    if bUseModel == bUseModelList[0]:
        pass
    elif bUseModel == bUseModelList[0]:
        config.set_flag(trt.BuilderFlag.FP16)
    elif bUseModel == bUseModelList[0]:
        config.set_flag(trt.BuilderFlag.INT8)
    else:
        raise AttributeError(f"{bUseModel} is not in {bUseModelList}!")
        # 2.设置最大的推理显存
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30) 

    if useSourceApi:
            # 1.设置网络的输入Tensor（只在batch上进行了设置，可以设置所有维度）
        inputTensor = network.add_input('inputT0', trt.float32, [-1, 1, nHeight, nWidth])
            # 2. 设置网络的输入Tensor的动态维度， 最小、最常用、最大
        profile.set_shape(inputTensor.name, [1, 1, nHeight, nWidth], [8, 1, nHeight, nWidth], [16, 1, nHeight, nWidth])
        # 完成网络的参数配置
        config.add_optimization_profile(profile)
        network = model_build(paraFile=paraFile, network=network, inputTensor=inputTensor)
    else:
        parser = trt.OnnxParser(network, logger)
        assert os.path.exists(onnxFile), f"{onnxFile} is not exists!"
        with open(onnxFile, 'rb') as model:
            if not parser.parse(model.read()):
                print("Failed parsing .onnx file!")
                for error in range(parser.num_errors):
                    print(parser.get_error(error))
                exit()
            print("Succeeded parsing .onnx file!")
        inputTensor = network.get_input(0)
        profile.set_shape(inputTensor.name, [1, 1, nHeight, nWidth], [8, 1, nHeight, nWidth], [16, 1, nHeight, nWidth])
        config.add_optimization_profile(profile)

        network.unmark_output(network.get_output(0))
    
    # 构建推理引擎序列化文件
    engineString = builder.build_serialized_network(network, config)

    # 将保存的推理引擎通过反序列化产生推理引擎
    engine = trt.Runtime(logger).deserialize_cuda_engine(engineString)

    ends = time.time()
    build_time = ends - start

    # 获取engine中输入输出Tensor的数量
    nIO = engine.num_io_tensors
    # 获取每个位置上Tensor的名字
    TensorNames = [engine.get_tensor_name(i) for i in range(nIO)]
    # 获取所有输入Tensor的数量
    nInput = [engine.get_tensor_mode(TensorNames[i]) for i in range(nIO)].count(trt.TensorIOMode.INPUT)

    # 通过CPU开辟内存，内存大小等于输入输出Tensor大小之和
    bufferH = []
    # 将输入数据转为推理引擎兼容的数据
    eval_data = Mydataset(MNIST(root='./lib', train=False, download=True))
    eval_loader = DataLoader(
        dataset=eval_data,
        batch_size=bs,
        shuffle=False,
        num_workers=n_w
    )
    start = time.time()
    for idx, (eval_x, eval_y) in enumerate(eval_loader):
        data = np.array(eval_x).reshape(-1, 1, nHeight, nWidth)
        # data = cv.imread(inferenceData, cv.IMREAD_GRAYSCALE).astype(np.float32).reshape(1, 1, nHeight, nWidth)
        # 转化为连续内存
        bufferH.append(np.ascontiguousarray(data))

        # context为GPU中的调度单元，设置推理时张量的真正形状
        context = engine.create_execution_context()
        context.set_input_shape(TensorNames[0], data.shape)

        # for i in range(nIO):
        #     print(f"[{i}] {'input' if i < nInput else 'output'}",
        #         f"\n  张量类型：{engine.get_tensor_dtype(TensorNames[i])}", 
        #         f"\n  张量形状：{engine.get_tensor_shape(TensorNames[i])}",
        #         f"\n  推理形状: {context.get_tensor_shape(TensorNames[i])}",
        #         f"\n  张量名：{TensorNames[i]}"
        #         )
            
        # 为输出Tensor通过CPU开辟内存
        for i in range(nInput, nIO):
            bufferH.append(
                np.empty(
                context.get_tensor_shape(TensorNames[i]), dtype=trt.nptype(engine.get_tensor_dtype(TensorNames[i]))
                )
            )
        
        bufferD = []
        for i in range(nIO):
            bufferD.append(
                cudart.cudaMalloc(
                bufferH[i].nbytes
                )[1]
            )
        
        for i in range(nInput):
            cudart.cudaMemcpy(
                bufferD[i], 
                bufferH[i].ctypes.data, 
                bufferH[i].nbytes, 
                cudart.cudaMemcpyKind.cudaMemcpyHostToDevice
            )
        # 执行推理
        context.execute_async_v3(0)

        for i in range(nInput, nIO):
            cudart.cudaMemcpy(bufferH[i].ctypes.data, 
                            bufferD[i], 
                            bufferH[i].nbytes, 
                            cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost)

        # for i in range(nIO):
        #     print(TensorNames[i])
        #     print(bufferH[i])
        for b in bufferD:
            cudart.cudaFree(b)
        break
    ends = time.time()
    endurance = ends - start
    print(f"build time:{round(build_time, precision)}秒, inference time:{round(endurance, precision)}秒")



if __name__ == "__main__":
    run()