import numpy as np
import torch
import torch.nn as nn
import FrEIA.framework as Ff
import FrEIA.modules as Fm


def build_starting_block(channels, n_nodes, condition, last_node, clamp=1.0, name="start"):
    def sub_conv(ch_hidden, kernel):
        pad = kernel // 2
        return lambda ch_in, ch_out: nn.Sequential(
            nn.Conv2d(ch_in, ch_hidden, kernel, padding=pad),
            nn.ReLU(),
            nn.Conv2d(ch_hidden, ch_out, kernel, padding=pad))

    subnet = sub_conv(channels, 3)
    block = [Ff.Node(last_node, Fm.GLOWCouplingBlock,
                             {'subnet_constructor': subnet, 'clamp': clamp},
                             conditions=condition, name=f"{name}_0")]
    for k in range(n_nodes-1):
        block.append(Ff.Node(block[-1], Fm.GLOWCouplingBlock,
                             {'subnet_constructor': subnet, 'clamp': clamp},
                             conditions=condition,
                             name=f"{name}_{k}"))
    block.append(Ff.Node(block[-1], Fm.HaarDownsampling, {'rebalance': 0.5}))

    return block


def build_block(channels, n_nodes, condition, last_node, clamp=1.0, split=(4, 4),
                            downsample="haar", name="block"):
    def sub_conv(ch_hidden, kernel):
        pad = kernel // 2
        return lambda ch_in, ch_out: nn.Sequential(
            nn.Conv2d(ch_in, ch_hidden, kernel, padding=pad),
            nn.ReLU(),
            nn.Conv2d(ch_hidden, ch_out, kernel, padding=pad))

    subnet = sub_conv(channels, 1)
    block = [Ff.Node(last_node, Fm.GLOWCouplingBlock,
                     {'subnet_constructor': subnet, 'clamp': clamp},
                     conditions=condition, name=f"{name}_0")]
    for k in range(n_nodes-1):
        subnet = sub_conv(channels, 3 if k % 2 else 1)
        block.append(Ff.Node(block[-1], Fm.GLOWCouplingBlock,
                             {'subnet_constructor': subnet, 'clamp': clamp},
                             conditions=condition, name=f"{name}_{k}"))
        block.append(Ff.Node(block[-1], Fm.PermuteRandom, {'seed': k}, name=f"{name}_{k}_permute"))

    # splitting parts of the image
    block.append(Ff.Node(block[-1], Fm.Split1D,
                         {'split_size_or_sections': split, 'dim': 0}, name=f"{name}_split"))
    split_node = Ff.Node(block[-1].out1, Fm.Flatten, {}, name=f"{name}_flat_splitted")

    if downsample == 'haar' or downsample == 1:
        block.append(Ff.Node(block[-1], Fm.HaarDownsampling, {'rebalance': 0.5}, name=f"{name}_haar_downsample"))

    # elif downsample == "checkerboard":
    #     block.append(Ff.Node(block[-1], Fm.i_revnet_downsampling, {}))

    return block, split_node


def build_fc_block(last_node, condition, n_blocks=4, subnet_width=512, clamp=0.6):
    def sub_fc(ch_hidden):
        return lambda ch_in, ch_out: nn.Sequential(
            nn.Linear(ch_in, ch_hidden),
            nn.ReLU(),
            nn.Linear(ch_hidden, ch_out))
    subnet = sub_fc(subnet_width)

    nodes = [Ff.Node(last_node, Fm.GLOWCouplingBlock,
                     {'subnet_constructor': subnet, 'clamp': clamp},
                     conditions=condition, name=f'fc_block_0'),

             ]
    for k in range(n_blocks-1):
        nodes.append(Ff.Node(nodes[-1], Fm.GLOWCouplingBlock,
                             {'subnet_constructor': subnet, 'clamp': clamp},
                             conditions=condition, name=f'fc_block_{k}'))
        nodes.append(Ff.Node(nodes[-1], Fm.PermuteRandom, {'seed': k}, name=f"fc_block_{k}_permute"))
    return nodes


def build_inn(data_res=(28, 28), cond_channels=(64, 128, 128, 512), subnet_conv_channels=(32, 64, 128),
              downsample=(1, 0), block_depth=4, splits=((2, 6), (4, 4))):
    # resolutions = []

    nodes = [Ff.InputNode(2, *data_res)]
    split_nodes = []
    # outputs of the cond. net at different resolution levels
    conditions = [Ff.ConditionNode(cond_channels[0], *data_res, name="c0")]
    conditions.extend( [Ff.ConditionNode(cond_channels[i], *[x / (2**(i)) for x in data_res], name="c1") for i in range(1, len(cond_channels)-1)])
    conditions.append(Ff.ConditionNode(cond_channels[-1], name="c3"))

    # conditions = [Ff.ConditionNode(cond_channels[0], *data_res, name="c0"),
    # Ff.ConditionNode(cond_channels[1], *[x / 2 for x in data_res], name="c1"),
    # Ff.ConditionNode(cond_channels[2], *[x / 4 for x in data_res], name="c2"),
    # Ff.ConditionNode(cond_channels[-1], name="c3"))
    print(len(conditions))

    block_0 = build_starting_block(subnet_conv_channels[0], 2, conditions[0], nodes[-1], clamp=1, name="start_block")
    nodes.extend(block_0)
    current_channels = 8
    for j in range(len(splits)):
        # if resolutions[j]

        from imageflow.utils import make_split
        split = make_split(current_channels, splits[j])
        block, split_node = build_block(subnet_conv_channels[j+1], block_depth, conditions[j+1], nodes[-1],
                                        clamp=1,
                                        split=split,
                                        name=f"block_{j}",
                                        downsample=downsample[j])
        nodes.extend(block)
        current_channels = split[0]
        current_channels *= 2**len(data_res)
        split_nodes.append(split_node)


    # block_1, split_node_1 = build_block(64, 4, conditions[1], nodes[-1], clamp=1, split=[2, 6], name="block_1", downsample="haar")
    # nodes.extend(block_1)
    # split_nodes.append(split_node_1)
    #
    # block_2, split_node_2 = build_block(128, 4, conditions[2], nodes[-1], clamp=0.6, split=[4, 4], name="block_2", downsample="none")
    # nodes.extend(block_2)
    # split_nodes.append(split_node_2)

    nodes.append(Ff.Node(nodes[-1], Fm.Flatten, {}, name='flatten'))

    fc_block = build_fc_block(nodes[-1], conditions[-1])
    nodes.extend(fc_block)

    nodes.append(Ff.Node([s.out0 for s in split_nodes] + [nodes[-1].out0],
                         Fm.Concat1d, {'dim': 0}, name="concat_node"))
    nodes.append(Ff.OutputNode(nodes[-1], name="output_node"))

    return Ff.ReversibleGraphNet(nodes + split_nodes + conditions, verbose=False)
#
# # data_res = (28, 28)
# data_res = (64, 64)
#
# # Testcase less channels
# print("-- Testing less channels -----")
# cond_net = MultiResCondNet(data_res=data_res, channels=(32, 64, 128))
# cond = cond_net(torch.randn(32, 2, *data_res))
# print(" * conditioning net output shapes:")
# [print(c.shape) for c in cond]
# cinn = build_inn(data_res=data_res, cond_channels=(32, 64, 128, 512))
# dat = torch.randn(32, 2, *data_res)
# res = cinn(dat, c=cond)
# print(f" * cinn output shape: {res.shape}")
#
#
# # Test case original
# print("-- Testing original case -----")
# cond_net = MultiResCondNet(data_res=data_res, channels=(64, 128, 256))
# cond = cond_net(torch.randn(32, 2, *data_res))
# print(" * conditioning net output shapes:")
# [print(c.shape) for c in cond]
# cinn = build_inn(data_res=data_res, cond_channels=(64, 128, 256, 512))
# dat = torch.randn(32, 2, *data_res)
# res = cinn(dat, c=cond)
# print(f" * cinn output shape: {res.shape}")
#
#
# # Test case more blocks
# print("-- Testing 4 blocks -----")
# cond_net = MultiResCondNet(data_res=data_res, channels=(64, 128, 128, 256))
# cond = cond_net(torch.randn(32, 2, *data_res))
# print(" * conditioning net output shapes:")
# [print(c.shape) for c in cond]
# cinn = build_inn(data_res=data_res, cond_channels=(64, 128, 128, 256, 512), splits=((2, 6), (4, 4), (8, 8)))
# dat = torch.randn(32, 2, *data_res)
# res = cinn(dat, c=cond)
# print(f" * cinn output shape: {res.shape}")
#
# # Test case deeper blocks
# cond_net = MultiResCondNet(data_res=data_res, channels=(64, 128, 128))
# cond = cond_net(torch.randn(32, 2, *data_res))
# print(" * conditioning net output shapes:")
# [print(c.shape) for c in cond]
# cinn = build_inn(data_res=data_res, cond_channels=(64, 128, 128, 512), splits=((2, 6), (4, 4)), block_depth=6)
# dat = torch.randn(32, 2, *data_res)
# res = cinn(dat, c=cond)



# cinn = build_inn(data_res=data_res, cond_channels=(64,128,256,512))
# dat = torch.randn(32, 2, *data_res)
# res = cinn(dat, c=cond)
# log_j = cinn.log_jacobian(run_forward=False)
# print(res.shape)

# cinn = build_inn(data_res=data_res, cond_channels=(64, 128, 128, 256, 512), splits=((2, 6), (4, 4), (8, 8)))
# dat = torch.randn(32, 2, *data_res)
# res = cinn(dat, c=cond)
# log_j = cinn.log_jacobian(run_forward=False)


def build_legacy_inn(self, data_res=(28, 28)):
    def sub_conv(ch_hidden, kernel):
        pad = kernel // 2
        return lambda ch_in, ch_out: nn.Sequential(
            nn.Conv2d(ch_in, ch_hidden, kernel, padding=pad),
            nn.ReLU(),
            nn.Conv2d(ch_hidden, ch_out, kernel, padding=pad))

    def sub_fc(ch_hidden):
        return lambda ch_in, ch_out: nn.Sequential(
            nn.Linear(ch_in, ch_hidden),
            nn.ReLU(),
            nn.Linear(ch_hidden, ch_out))

    nodes = [Ff.InputNode(2, *data_res)]
    # outputs of the cond. net at different resolution levels
    conditions = [Ff.ConditionNode(64, *data_res),
                  Ff.ConditionNode(128, *[x / 2 for x in data_res]),
                  Ff.ConditionNode(128, *[x / 4 for x in data_res]),
                  Ff.ConditionNode(512)]

    split_nodes = []

    subnet = sub_conv(32, 3)
    for k in range(2):
        nodes.append(Ff.Node(nodes[-1], Fm.GLOWCouplingBlock,
                             {'subnet_constructor': subnet, 'clamp': 1.0},
                             conditions=conditions[0]))

    nodes.append(Ff.Node(nodes[-1], Fm.HaarDownsampling, {'rebalance': 0.5}))

    for k in range(4):
        subnet = sub_conv(64, 3 if k % 2 else 1)

        nodes.append(Ff.Node(nodes[-1], Fm.GLOWCouplingBlock,
                             {'subnet_constructor': subnet, 'clamp': 1.0},
                             conditions=conditions[1]))
        nodes.append(Ff.Node(nodes[-1], Fm.PermuteRandom, {'seed': k}))

    # split off 6/8 ch
    nodes.append(Ff.Node(nodes[-1], Fm.Split1D,
                         {'split_size_or_sections': [2, 6], 'dim': 0}))
    split_nodes.append(Ff.Node(nodes[-1].out1, Fm.Flatten, {}))

    nodes.append(Ff.Node(nodes[-1], Fm.HaarDownsampling, {'rebalance': 0.5}))

    for k in range(4):
        subnet = sub_conv(128, 3 if k % 2 else 1)

        nodes.append(Ff.Node(nodes[-1], Fm.GLOWCouplingBlock,
                             {'subnet_constructor': subnet, 'clamp': 0.6},
                             conditions=conditions[2]))
        nodes.append(Ff.Node(nodes[-1], Fm.PermuteRandom, {'seed': k}))

    # split off 4/8 ch
    nodes.append(Ff.Node(nodes[-1], Fm.Split1D,
                         {'split_size_or_sections': [4, 4], 'dim': 0}))
    split_nodes.append(Ff.Node(nodes[-1].out1, Fm.Flatten, {}))
    nodes.append(Ff.Node(nodes[-1], Fm.Flatten, {}, name='flatten'))

    # fully_connected part
    subnet = sub_fc(512)
    for k in range(4):
        nodes.append(Ff.Node(nodes[-1], Fm.GLOWCouplingBlock,
                             {'subnet_constructor': subnet, 'clamp': 0.6},
                             conditions=conditions[3]))
        nodes.append(Ff.Node(nodes[-1], Fm.PermuteRandom, {'seed': k}))

    # concat everything
    nodes.append(Ff.Node([s.out0 for s in split_nodes] + [nodes[-1].out0],
                         Fm.Concat1d, {'dim': 0}))
    nodes.append(Ff.OutputNode(nodes[-1]))

    return Ff.ReversibleGraphNet(nodes + split_nodes + conditions, verbose=False)