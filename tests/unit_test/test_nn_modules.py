# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
kernels = [
    "ConvBnRelu",
    "ConvBnRelu6",
    "ConvBn",
    "ConvRelu",
    "ConvRelu6",
    "ConvHswish",
    "ConvBlock",
    "ConvBnHswish",
    "DwConvBn",
    "DwConvRelu",
    "DwConvRelu6",
    "DwConvBnRelu",
    "DwConvBnRelu6",
    "DwConvBlock",
    "ConvBnHswish",
    "MaxPoolBlock",
    "AvgPoolBlock",
    "FCBlock",
    "ConcatBlock",
    "SplitBlock",
    "ChannelShuffle",
    "SEBlock",
    "GlobalAvgPoolBlock",
    "BnRelu",
    "BnBlock",
    "HswishBlock",
    "ReluBlock",
    "AddRelu",
    "AddBlock"]

if __name__ == '__main__':
    config = {
        "HW": 24,
        "CIN": 144,
        "COUT": 32,
        "KERNEL_SIZE": 1,
        "STRIDES": 1,
        "POOL_STRIDES": 2,
        "NS": 3,
        "CIN1": 12,
        "CIN2": 12,
        "CIN3": 12,
        "CIN4": 12
    }
    # test tersorflow kernels, tensorflow==2.7.0 or 2.6.0 is needed
    from nn_meter.builder.nn_modules.tf_networks import blocks
    for kernel in kernels:
        getattr(blocks, kernel)(config).test_block()

    # test torch kernels, torch==1.10.0 or 1.9.0 is needed
    from nn_meter.builder.nn_modules.torch_networks import blocks
    for kernel in kernels:
        getattr(blocks, kernel)(config).test_block()
