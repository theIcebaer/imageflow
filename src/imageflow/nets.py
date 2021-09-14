import os
import torch
import torch.nn as nn

import FrEIA.framework as Ff
import FrEIA.modules as Fm

os.environ['VXM_BACKEND'] = 'pytorch'
from voxelmorph.torch.layers import VecInt
from voxelmorph.torch.layers import SpatialTransformer


class Reg_mnist_cINN(nn.Module):
    def __init__(self, device=torch.device("cpu")):
        super(Reg_mnist_cINN, self).__init__()
        self.flat_layer = torch.nn.Flatten(start_dim=1, end_dim=-1)
        self.cinn = self.build_inn()
        for p in self.cinn.parameters():
            if p.requires_grad:
                p.data = 0.01 * torch.randn_like(p)
        self.device = device

        self.integrator = VecInt(inshape=(28, 28), nsteps=7).to(device)
        self.transformer = SpatialTransformer(size=(28, 28)).to(device)

    def build_inn(self):

            def subnet(ch_in, ch_out):
                return nn.Sequential(nn.Linear(ch_in, 512),
                                    nn.ReLU(),
                                    nn.Linear(512, ch_out))

            cond = Ff.ConditionNode(2 * 28 * 28, name='condition')
            nodes = [Ff.InputNode(2, 28, 28, name='flat')]

            nodes.append(Ff.Node(nodes[-1], Fm.Flatten, {}, name='flat'))

            for k in range(5):
                nodes.append(Ff.Node(nodes[-1], Fm.PermuteRandom , {'seed':k}, name="permute_{}".format(k)))
                nodes.append(Ff.Node(nodes[-1], Fm.GLOWCouplingBlock,
                                    {'subnet_constructor':subnet, 'clamp':1.0},
                                    conditions=cond,
                                    name="Coupling_Block_{}".format(k)))

            return Ff.ReversibleGraphNet(nodes + [cond, Ff.OutputNode(nodes[-1])], verbose=False)

    def forward(self, x, c, rev=False):
        z = self.cinn(x, [self.flat_layer(c)], rev=rev)
        log_j = self.cinn.log_jacobian(run_forward=False)

        return z, log_j

    def reverse_sample(self, z, cond):
        v_field = self.cinn(z, c=[self.flat_layer(cond)], rev=True)
        logj_rev = self.cinn.log_jacobian(run_forward=False)
        source = cond[:, :1, ...].to(self.device)
        target = cond[:, 1:, ...].to(self.device)

        deformation = self.integrator(v_field)
        target_pred = self.transformer(source, deformation)

        return target_pred, v_field, logj_rev


class MultiresCondNet(nn.Module):

    def __init__(self):
        super().__init__()

        class Flatten(nn.Module):
            def __init__(self, *args):
                super().__init__()
            def forward(self, x):
                return x.view(x.shape[0], -1)

        # self.Flatten = Flatten()
        # self.Linear = nn.Linear(512,512)

        self.resolution_levels = nn.ModuleList([
                                                nn.Sequential(nn.Conv2d(2, 64, 3, padding=1), nn.LeakyReLU(), nn.Conv2d(64, 64, 3, padding=1)),
                                                nn.Sequential(nn.LeakyReLU(), nn.Conv2d(64, 128, 3, padding=1), nn.LeakyReLU(), nn.Conv2d(128,128,3, padding=1, stride=2)),
                                                nn.Sequential(nn.LeakyReLU(),nn.Conv2d(128, 128, 3, padding=1, stride=2)),
                                                nn.Sequential(nn.LeakyReLU(), nn.AvgPool2d(3), Flatten(), nn.Linear(512, 512))
                                                ])

    def forward(self, c):
        outputs = [c]
        # print(c.shape)
        for i, m in enumerate(self.resolution_levels):
            # print(i)
            # print(m.shape)
            outputs.append(m(outputs[-1]))
        #     print(outputs[-1].shape)
        # outputs[-1] = nn.LeakyReLU()(outputs[-1])
        # print(outputs[-1].shape)
        # outputs[-1] = nn.AvgPool2d(3)(outputs[-1])
        # print(outputs[-1].shape)
        # outputs[-1] = self.Flatten(outputs[-1])
        # print(outputs[-1].shape)
        # outputs[-1] =  self.Linear(outputs[-1])
        return outputs[1:]


class CinnConvMultires(nn.Module):
    """
    Structure from ardizzone et al. on conditional INNs TODO:ref
    """
    def __init__(self):
        super(CinnConvMultires, self).__init__()
        self.cinn = self.build_inn()
        self.cond_net = MultiresCondNet()

    def build_inn(self):

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

        nodes = [Ff.InputNode(2, 28, 28)]
        # outputs of the cond. net at different resolution levels
        conditions = [Ff.ConditionNode(64, 28, 28),
                      Ff.ConditionNode(128, 14, 14),
                      Ff.ConditionNode(128, 7, 7),
                      Ff.ConditionNode(512)]

        split_nodes = []

        subnet = sub_conv(32, 3)
        for k in range(2):
            nodes.append(Ff.Node(nodes[-1], Fm.GLOWCouplingBlock,
                                 {'subnet_constructor':subnet, 'clamp':1.0},
                                 conditions=conditions[0]))

        nodes.append(Ff.Node(nodes[-1], Fm.HaarDownsampling, {'rebalance':0.5}))

        for k in range(4):
            subnet = sub_conv(64, 3 if k%2 else 1)

            nodes.append(Ff.Node(nodes[-1], Fm.GLOWCouplingBlock,
                                 {'subnet_constructor':subnet, 'clamp':1.0},
                                 conditions=conditions[1]))
            nodes.append(Ff.Node(nodes[-1], Fm.PermuteRandom, {'seed':k}))

        #split off 6/8 ch
        nodes.append(Ff.Node(nodes[-1], Fm.Split1D,
                             {'split_size_or_sections':[2,6], 'dim':0}))
        split_nodes.append(Ff.Node(nodes[-1].out1, Fm.Flatten, {}))

        nodes.append(Ff.Node(nodes[-1], Fm.HaarDownsampling, {'rebalance':0.5}))

        for k in range(4):
            subnet = sub_conv(128, 3 if k%2 else 1)

            nodes.append(Ff.Node(nodes[-1], Fm.GLOWCouplingBlock,
                                 {'subnet_constructor':subnet, 'clamp':0.6},
                                 conditions=conditions[2]))
            nodes.append(Ff.Node(nodes[-1], Fm.PermuteRandom, {'seed':k}))

        #split off 4/8 ch
        nodes.append(Ff.Node(nodes[-1], Fm.Split1D,
                             {'split_size_or_sections':[4,4], 'dim':0}))
        split_nodes.append(Ff.Node(nodes[-1].out1, Fm.Flatten, {}))
        nodes.append(Ff.Node(nodes[-1], Fm.Flatten, {}, name='flatten'))

        # fully_connected part
        subnet = sub_fc(512)
        for k in range(4):
            nodes.append(Ff.Node(nodes[-1], Fm.GLOWCouplingBlock,
                                 {'subnet_constructor':subnet, 'clamp':0.6},
                                 conditions=conditions[3]))
            nodes.append(Ff.Node(nodes[-1], Fm.PermuteRandom, {'seed':k}))

        # concat everything
        nodes.append(Ff.Node([s.out0 for s in split_nodes] + [nodes[-1].out0],
                             Fm.Concat1d, {'dim':0}))
        nodes.append(Ff.OutputNode(nodes[-1]))

        return Ff.ReversibleGraphNet(nodes + split_nodes + conditions, verbose=False)

    def forward(self, x, c, rev=False):
        c = self.cond_net(c)
        z = self.cinn(x, c, rev=rev)
        log_j = self.cinn.log_jacobian(run_forward=False)
        return z, log_j

    def reverse_sample(self, z, L):
        return self.cinn(z, c=self.cond_net(L), rev=True)

    def init(self):
        for p in self.cinn.parameters():
            if p.requires_grad:
                p.data = 0.01 * torch.randn_like(p)


class CinnBasic(nn.Module):
    """
    Class for supervised regis
    """
    def __init__(self, cond_net):
        super(CinnBasic, self).__init__()
        self.flat_layer = torch.nn.Flatten(start_dim=1, end_dim=-1)
        self.cinn = self.build_inn()
        self.cond_net = cond_net

    def build_inn(self):

        def subnet(ch_in, ch_out):
            return nn.Sequential(nn.Linear(ch_in, 512), nn.ReLU(),nn.Linear(512, ch_out))

        cond = Ff.ConditionNode(512, name='condition')

        nodes = [Ff.InputNode(2, 28, 28, name='flat')]

        nodes.append(Ff.Node(nodes[-1], Fm.Flatten, {}, name='flat'))

        for k in range(20):
            nodes.append(Ff.Node(nodes[-1], Fm.PermuteRandom , {'seed':k}, name="permute_{}".format(k)))
            nodes.append(Ff.Node(nodes[-1], Fm.GLOWCouplingBlock,
                                {'subnet_constructor':subnet, 'clamp':1.0},
                                conditions=cond,
                                name="Coupling_Block_{}".format(k)))

        return Ff.ReversibleGraphNet(nodes + [cond, Ff.OutputNode(nodes[-1])], verbose=False)

    def forward(self, x, c, rev=False):
        c = self.cond_net(c)
        z = self.cinn(x, [self.flat_layer(c)], rev=rev)
        log_j = self.cinn.log_jacobian(run_forward=False)
        return z, log_j

    def reverse_sample(self, z, c):
        # x = self.cinn((x, [self.flat_layer(c)], rev=True))
        # return x
        pass
    def init(self):
        for p in self.cinn.parameters():
            if p.requires_grad:
                p.data = 0.01 * torch.randn_like(p)


class CondNetWrapper(nn.Module):
    def __init__(self, cond_net, type=None):
        super().__init__()
        self.upscale = nn.Conv2d(2, 3, 3, padding=1)  # upscale the 2d-like image to a 3d input for the predefined network
        if type == "resnet":
            self.features = nn.Sequential(*list(cond_net.children())[:-2])
        elif type == "mobilenet":
            self.features = cond_net.features

    def forward(self, x):
        x = self.upscale(x)
        x = self.features(x)
        return x
