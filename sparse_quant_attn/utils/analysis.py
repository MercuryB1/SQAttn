import torch


def compute_speed_up(bits_alloc):
    # TODO speedup data need to modify
    BIT8_FLOPS = 400.0
    BIT4_FLOPS = 459.0
    FA2_FLOPS = 161.0

    all_bit8_values = []
    all_bit4_values = []

    # 遍历字典收集数据
    for key, value in bits_alloc.items():
        if 'bit8' in value:
            all_bit8_values.extend(value['bit8'])
        if 'bit4' in value:
            all_bit4_values.extend(value['bit4'])

    # 计算平均值
    bit8_mean = sum(all_bit8_values) / len(all_bit8_values)
    bit4_mean = sum(all_bit4_values) / len(all_bit4_values)
    sparse_mean = 1 - bit4_mean
    bit4_mean = bit4_mean - bit8_mean
    time_baseline = 1.0 / FA2_FLOPS
    time_our = bit8_mean / BIT8_FLOPS + bit4_mean / BIT4_FLOPS
    # speedup = bit8_mean * BIT8_SPEEDUP + bit4_mean * BIT4_SPEEDUP + sparse_mean * SPARSE_SPEEDUP
    speedup = time_baseline / time_our
    average_bits = bit8_mean * 8 + bit4_mean * 4
    return speedup, average_bits


