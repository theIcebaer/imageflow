import math
import os
import torch
import torch.nn as nn
import numpy as np

import FrEIA.framework as Ff
import FrEIA.modules as Fm

os.environ['VXM_BACKEND'] = 'pytorch'
from voxelmorph.torch.layers import VecInt
from voxelmorph.torch.layers import SpatialTransformer

from torchvision.models import mobilenet_v3_large
from torchvision.models import resnet18
from imageflow.inn_builder import build_inn


class Reg_mnist_cINN(nn.Module):
    def __init__(self, device, init_method="gaussian", init_params=None):
        super(Reg_mnist_cINN, self).__init__()
        self.flat_layer = torch.nn.Flatten(start_dim=1, end_dim=-1)
        self.cinn = self.build_inn()
        self.init_weights(method=init_method, init_args=init_params)

        self.device = device

        self.integrator = VecInt(inshape=(28, 28), nsteps=7).to(device)
        self.transformer = SpatialTransformer(size=(28, 28)).to(device)

    def init_weights(self, method="gaussian", init_args=None):
        if method == "gaussian":
            for p in self.cinn.parameters():
                if p.requires_grad:
                    if init_args is None:
                        p.data = 0.01 * torch.randn_like(p)
                    else:
                        p.data = init_args['lr'] * torch.randn_like(p)

        elif method == "xavier":
            for p in self.cinn.parameters():
                if p.requires_grad:
                    if init_args is None:
                        torch.nn.init.xavier_uniform_(p, gain=1.0)
                    else:
                        torch.nn.init.xavier_uniform_(p, gain=init_args['gain'])

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
            nodes.append(Ff.Node(nodes[-1],
                                 Fm.GLOWCouplingBlock,
                                 {'subnet_constructor': subnet, 'clamp': 1.0},
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


class MultiResCondNet(nn.Module):

    def __init__(self, data_res=(28, 28), channels=(64, 128, 128)):
        super().__init__()

        class Flatten(nn.Module):
            def __init__(self, *args):
                super().__init__()
            def forward(self, x):
                return x.view(x.shape[0], -1)

        def _get_lin_res(data_res, kernel_size):
            h_in = np.array([x/ (2**(len(channels)-1)) for x in data_res])
            h_out = np.floor((h_in-kernel_size)/kernel_size + 1)
            lin_shape = int(channels[-1] * h_out[0] * h_out[1])
            return lin_shape

        avg_pool_kernel_size = 3 if data_res[0] < 64 else 4

        lin_shape = _get_lin_res(data_res, avg_pool_kernel_size)

        # self.resolution_levels = nn.ModuleList([
        #     nn.Sequential(nn.Conv2d(2, channels[0], 3, padding=1), nn.LeakyReLU(), nn.Conv2d(64, 64, 3, padding=1)),
        #     nn.Sequential(nn.LeakyReLU(), nn.Conv2d(channels[0], channels[1], 3, padding=1), nn.LeakyReLU(),
        #                   nn.Conv2d(channels[1], channels[1], 3, padding=1, stride=2)),
        #     nn.Sequential(nn.LeakyReLU(), nn.Conv2d(channels[1], channels[2], 3, padding=1, stride=2)),
        #     nn.Sequential(nn.LeakyReLU(), nn.AvgPool2d(kernel_size=avg_pool_kernel_size), Flatten(), nn.Linear(lin_shape, 512))
        #     ])
        self.resolution_levels = nn.ModuleList([nn.Sequential(nn.Conv2d(2, channels[0], 3, padding=1), nn.LeakyReLU(),
                                                              nn.Conv2d(channels[0], channels[0], 3, padding=1))])
        for i in range(len(channels)-2):
            self.resolution_levels.append(
                nn.Sequential(nn.LeakyReLU(), nn.Conv2d(channels[i], channels[i+1], 3, padding=1), nn.LeakyReLU(),
                              nn.Conv2d(channels[i+1], channels[i+1], 3, padding=1, stride=2)))
        self.resolution_levels.append(nn.Sequential(nn.LeakyReLU(), nn.Conv2d(channels[-2], channels[-1], 3, padding=1, stride=2)))
        self.resolution_levels.append(nn.Sequential(nn.LeakyReLU(), nn.AvgPool2d(kernel_size=avg_pool_kernel_size), Flatten(), nn.Linear(lin_shape, 512)))

    def forward(self, c):
        outputs = [c]
        for i, m in enumerate(self.resolution_levels):
            outputs.append(m(outputs[-1]))
        return outputs[1:]


class CinnConvMultiRes(nn.Module):
    """
    Structure from ardizzone et al. on conditional INNs TODO:ref
    """
    def __init__(self, device=None, init_method="xavier", data_res=(28, 28), cond_channels=(64, 128, 128),
                 block_depth=4, splits=((2, 6), (4, 4)), downsample=(1, 0), conv_channels=(32, 64, 128), image_res=None):
        super(CinnConvMultiRes, self).__init__()
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.cinn = build_inn(data_res=data_res,
                              cond_channels=(*cond_channels, 512),
                              block_depth=block_depth,
                              splits=splits,
                              downsample=downsample,
                              subnet_conv_channels=conv_channels)
        if image_res:
            from imageflow.extra_nets import ScalingCondNet
            self.cond_net = ScalingCondNet(data_res=data_res, channels=cond_channels)
        else:
            self.cond_net = MultiResCondNet(data_res=data_res, channels=cond_channels)
        self.init_weights(init_method)

        self.integrator = VecInt(inshape=image_res, nsteps=7).to(device) if image_res else VecInt(inshape=data_res, nsteps=7).to(device)
        self.transformer = SpatialTransformer(size=image_res).to(device) if image_res else SpatialTransformer(size=data_res).to(device)
        self.model_architecture = {
            'data_res': data_res,
            'cond_channels': cond_channels,
            'block_depth': block_depth,
            'splits': splits,
            'downsample': downsample,
            'conv_channels': conv_channels
        }
        # maybe add pretraining here.
        if image_res:
            from voxelmorph.torch.layers import ResizeTransform
            factor = data_res[0] / image_res[0]
            self.resize_layer = ResizeTransform(factor, 2)
            self.inverse_resize_layer = ResizeTransform(1/factor, 2)
        else:
            self.resize_layer = None
            self.inverse_resize_layer = None

    def forward(self, x, c, rev=False):
        c = self.cond_net(c)
        if self.inverse_resize_layer:
            x = self.inverse_resize_layer(x)
        z = self.cinn(x, c, rev=rev)
        log_j = self.cinn.log_jacobian(run_forward=False)
        return z, log_j

    def reverse_sample(self, z, c, give_target=True, rescale_field=False):
        cond = self.cond_net(c)
        v_field = self.cinn(z, c=cond, rev=True)
        logj_rev = self.cinn.log_jacobian(run_forward=False)
        source = c[:, :1, ...].to(self.device)

        if self.resize_layer:
            v_field = self.resize_layer(v_field)


        if give_target:

            deformation = self.integrator(v_field)
            target_pred = self.transformer(source, deformation)

            return target_pred, v_field, logj_rev



        else:
            return v_field, logj_rev

    # def init_weights(self):
    #     for p in self.cinn.parameters():
    #         if p.requires_grad:
    #             p.data = 0.01 * torch.randn_like(p)
    def init_weights(self, method="gaussian"):
        """ This function does control the behaviour of the default weight initialization, by a set of keywords,
        currently being:
            - "gaussian" - for gaussian weight init
            - "xavier" - for weight init with the xavier-normal strategy
            - "uniform" - for uniform weights with the default pytorch mechanism. This is just default behaviour its
                          just an explicit tag for "stick to the default".
                          NOTE: Don't use this, for some reason this leads to incredibly big starting values for the

        """

        if method == "gaussian":
            def _init_gaussian(m):
                if isinstance(m, nn.Linear):
                    # torch.nn.init.normal_(m.weight, mean=0, std=1)
                    torch.nn.init.normal_(m.weight, std=0.01)
                    torch.nn.init.normal_(m.bias, std=0.01)

            self.apply(_init_gaussian)

            # for p in self.cinn.parameters():
            #     if p.requires_grad:
            #        p.data = 0.01 * torch.randn_like(p)

        elif method == "xavier":
            def _init_xavier(m):
                if isinstance(m, nn.Linear):
                    torch.nn.init.xavier_normal_(m.weight.data, gain=0.01)
                    torch.nn.init.constant_(m.bias.data, 0.0)
            self.apply(_init_xavier)

            def _init_last(m):
                if isinstance(m, nn.Linear):
                    torch.nn.init.constant_(m.weight.data, 0.0)
                    torch.nn.init.constant_(m.bias.data, 0.0)
            children = list(self.cinn.children())[0]
            last = children[-10]
            last.apply(_init_last)

        elif method == "uniform":
            pass
            # layers ar initialized by default from a uniform distribution, see:
            # https://pytorch.org/docs/stable/generated/torch.nn.Linear.html#torch.nn.Linear


class CinnBasic(nn.Module):
    def __init__(self, device=None, cond_net=None, init_method="gaussian", init_params=None, data_res=(28, 28), **kwargs):
        super(CinnBasic, self).__init__()

        if device is None:
            device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.device = device

        if cond_net is None:
            out_shape = (2*math.prod(data_res),)
        else:
            out_shape = cond_net.out_shape

        args = [arg for arg in ['subnet_depth', 'subnet_width', 'model_depth'] if arg in kwargs.keys()]
        subnet_args = {key: kwargs[key] for key in args}

        self.model_architecture = subnet_args


        # subnet_args = {}
        # if kwargs['subnet_depth']:
        #     subnet_args['subnet_depth'] = kwargs['subnet_depth']
        # if kwargs['subnet_width']:
        #     subnet_args['subnet_width'] = kwargs['subnet_width']
        # if kwargs['model_depth']:
        #     subnet_args['model_depth'] = kwargs['model_depth']

        self.flat_layer = torch.nn.Flatten(start_dim=1, end_dim=-1)

        if subnet_args:
            self.cinn = self.build_inn(out_shape, **subnet_args)
        else:
            self.cinn = self.build_inn(out_shape)
        self.cinn.to(self.device)

        # Note: Init cinn weights before integration of conditioning net to avoid rewriting cond weights.
        self.init_weights(method=init_method, init_params=init_params)

        self.cond_net = cond_net
        if self.cond_net is not None:
            self.cond_net = cond_net.to(device)
        # print(self.cond_net)

        self.integrator = VecInt(inshape=data_res, nsteps=7).to(self.device)
        self.transformer = SpatialTransformer(size=data_res).to(self.device)

    def init_weights(self, method="gaussian", init_params=None):
        """ This function does control the behaviour of the default weight initialization, by a set of keywords,
        currently being:
            - "gaussian" - for gaussian weight init
            - "xavier" - for weight init with the xavier-normal strategy
            - "uniform" - for uniform weights with the default pytorch mechanism. This is just default behaviour its
                          just an explicit tag for "stick to the default".
                          NOTE: Don't use this, for some reason this leads to incredibly big starting values for the

        """
        if method == "gaussian":
            def _init_gaussian(m):
                if init_params:
                    std = init_params['std']
                else:
                    std = 0.01
                if isinstance(m, nn.Linear):
                    # torch.nn.init.normal_(m.weight, mean=0, std=1)
                    torch.nn.init.normal_(m.weight, std=std)
                    torch.nn.init.normal_(m.bias, std=std)

            self.apply(_init_gaussian)

            # for p in self.cinn.parameters():
            #     if p.requires_grad:
            #        p.data = 0.01 * torch.randn_like(p)

        elif method == "xavier":
            if init_params:
                gain = init_params['gain']
            else:
                gain = 0.01

            def _init_xavier(m):
                if isinstance(m, nn.Linear):
                    torch.nn.init.xavier_normal_(m.weight.data, gain=gain)
                    torch.nn.init.constant_(m.bias.data, 0.0)
            self.apply(_init_xavier)

            def _init_last(m):
                if isinstance(m, nn.Linear):
                    torch.nn.init.constant_(m.weight.data, 0.0)
                    torch.nn.init.constant_(m.bias.data, 0.0)
            children = list(self.cinn.children())[0]
            last = children[-3]
            last.apply(_init_last)

        elif method == "xavier_uniform":
            if init_params:
                gain = init_params['gain']
            else:
                gain = 0.01

            def _init_xavier_uniform(m):
                if isinstance(m, nn.Linear):
                    torch.nn.init.xavier_uniform_(m.weight.data, gain=gain)
                    torch.nn.init.constant_(m.bias.data, 0.0)
            self.apply(_init_xavier_uniform)

            def _init_last(m):
                if isinstance(m, nn.Linear):
                    torch.nn.init.constant_(m.weight.data, 0.0)
                    torch.nn.init.constant_(m.bias.data, 0.0)
            children = list(self.cinn.children())[0]
            last = children[-3]
            last.apply(_init_last)


        elif method == 'kaiming':
            def _init_kaiming(m):
                if isinstance(m, nn.Linear):
                    torch.nn.init.kaiming_normal_(m.weight.data, nonlinearity='relu')
                    torch.nn.init.constant_(m.bias.data, 0.0)
            self.apply(_init_kaiming)

        elif method == "kaiming_uniform":
            def _init_kaiming_uniform(m):
                if isinstance(m, nn.Linear):
                    torch.nn.init.kaiming_uniform_(m.weight.data, nonlinearity='relu')
                    torch.nn.init.constant_(m.bias.data, 0.0)
            self.apply(_init_kaiming_uniform)

        elif method == "uniform":
            pass
            # layers ar initialized by default from a uniform distribution, see:
            # https://pytorch.org/docs/stable/generated/torch.nn.Linear.html#torch.nn.Linear

    def build_inn(self, cond_shape, subnet_width=512, subnet_depth=1, model_depth=20):

        def subnet(ch_in, ch_out):
            modules = []
            modules.append(nn.Linear(ch_in, subnet_width))
            modules.append(nn.ReLU())
            for _ in range(subnet_depth-1):
                modules.append(nn.Linear(subnet_width, subnet_width))
                modules.append(nn.ReLU())
            modules.append(nn.Linear(subnet_width, ch_out))

            return nn.Sequential(*modules)

        cond = Ff.ConditionNode(*cond_shape, name='condition')
        nodes = [Ff.InputNode(2, 28, 28, name='flat')]
        nodes.append(Ff.Node(nodes[-1], Fm.Flatten, {}, name='flat'))

        for k in range(model_depth):
            nodes.append(Ff.Node(nodes[-1], Fm.PermuteRandom , {'seed': k}, name="permute_{}".format(k)))
            nodes.append(Ff.Node(nodes[-1],
                                 Fm.GLOWCouplingBlock,
                                 {'subnet_constructor': subnet, 'clamp': 1.0},
                                 conditions=cond,
                                 name="Coupling_Block_{}".format(k)))

        return Ff.ReversibleGraphNet(nodes + [cond, Ff.OutputNode(nodes[-1])], verbose=False)

    def reverse_sample(self, z, cond):
        if self.cond_net:
            cond = self.cond_net(cond)
        v_field = self.cinn(z, c=[self.flat_layer(cond)], rev=True)
        logj_rev = self.cinn.log_jacobian(run_forward=False)
        source = cond[:, :1, ...].to(self.device)
        # target = cond[:, 1:, ...].to(self.device)

        deformation = self.integrator(v_field)
        target_pred = self.transformer(source, deformation)

        return target_pred, v_field, logj_rev

    def forward(self, x, c, rev=False):
        if self.cond_net:
            c = self.cond_net(c)
        c = [self.flat_layer(c)]
        z = self.cinn(x, c, rev=rev)
        log_j = self.cinn.log_jacobian(run_forward=False)
        return z, log_j


class CondNetWrapper(nn.Module):

    def __init__(self, net_type=None, pretraining="fixed"):
        super().__init__()
        # upscale the 2 channel image to a 3 channel input for the predefined network
        self.upscale = nn.Conv2d(2, 3, 3, padding=1)
        self.net_type = net_type
        self.pretraining = pretraining
        if net_type == "resnet":
            # TODO: make differentiation depending on data shape here for future data formats.
            self.out_shape = (512,)
            if pretraining == "fixed":
                print(f"Creating conditioning network {net_type} with fixed pretraining ({pretraining})")
                cond_net = resnet18(pretrained=True)
                self.features = nn.Sequential(*list(cond_net.children())[:-2])
                for param in self.features:
                    param.requires_grad_(False)

            elif pretraining == "fine_tune_last":
                print(f"Creating conditioning network {net_type} with fine_tune_last pretraining ({pretraining})")
                cond_net = resnet18(pretrained=True)
                self.features = nn.Sequential(*list(cond_net.children())[:-2])
                for param in self.features[:-1]:
                    param.requires_grad_(False)

            elif pretraining == "fine_tune_full":
                print(f"Creating conditioning network {net_type} with fine_tune_full pretraining ({pretraining})")
                cond_net = resnet18(pretrained=True)
                self.features = nn.Sequential(*list(cond_net.children())[:-2])

            else:  # No pretraining
                print(f"Creating conditioning network {net_type} with no pretraining ({pretraining})")
                cond_net = resnet18(pretrained=False)
                self.features = nn.Sequential(*list(cond_net.children())[:-2])

        elif net_type == "mobilenet":

            if pretraining == 'fixed':
                cond_net = mobilenet_v3_large(pretrained=True)
                self.features = cond_net.features
                for param in self.features:
                    param.requires_grad_(False)

            elif pretraining == 'fine_tune_last':
                cond_net = mobilenet_v3_large(pretrained=True)
                self.features = cond_net.features
                for param in self.features[:-4]:
                    param.requires_grad_(False)

            elif pretraining == 'fine_tune_full':
                cond_net = mobilenet_v3_large(pretrained=True)
                self.features = cond_net.features

            else:  # Not pretraining
                cond_net = mobilenet_v3_large(pretrained=False)
                self.features = cond_net.features

            # TODO: make differentiation depending on data shape here for future data formats.
            self.out_shape = (960,)

        else:
            print("No conditioning network specified.")

    # def out_shape(self):
    #     return self.out_shape

    def forward(self, x):
        x = self.upscale(x)
        x = self.features(x)
        # print(x.shape)
        return x


class ResidualFlow(nn.Module):


    pass


class CouplingFlow(nn.Module):
    def __init__(self, device=None, cond_net=None, init_method="gaussian", init_params=None):
        super(CouplingFlow, self).__init__()

        if device is None:
            device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.device = device

        if cond_net is None:
            out_shape = (1*28*28,)
        else:
            out_shape = cond_net.out_shape

        self.flat_layer = torch.nn.Flatten(start_dim=1, end_dim=-1)
        self.cinn = self.build_inn(out_shape)
        self.cinn.to(self.device)

        # Note: Init cinn weights before integration of conditioning net to avoid rewriting cond weights.
        self.init_weights(method=init_method)

        self.cond_net = cond_net
        if self.cond_net is not None:
            self.cond_net = cond_net.to(device)

        self.integrator = VecInt(inshape=(28, 28), nsteps=7).to(self.device)
        self.transformer = SpatialTransformer(size=(28, 28)).to(self.device)

    def build_inn(self, cond_shape):

        def subnet(ch_in, ch_out):
            return nn.Sequential(nn.Linear(ch_in, 512),
                                 nn.ReLU(),
                                 nn.Linear(512, ch_out))

        cond = Ff.ConditionNode(*cond_shape, name='condition')
        nodes = [Ff.InputNode(1, 28, 28, name='flat')]
        nodes.append(Ff.Node(nodes[-1], Fm.Flatten, {}, name='flat'))

        for k in range(20):
            nodes.append(Ff.Node(nodes[-1], Fm.PermuteRandom , {'seed': k}, name="permute_{}".format(k)))
            nodes.append(Ff.Node(nodes[-1],
                                 Fm.GLOWCouplingBlock,
                                 {'subnet_constructor': subnet, 'clamp': 1.0},
                                 conditions=cond,
                                 name="Coupling_Block_{}".format(k)))

        return Ff.ReversibleGraphNet(nodes + [cond, Ff.OutputNode(nodes[-1])], verbose=False)

    def reverse_sample(self, z, cond):
        if self.cond_net:
            cond = self.cond_net(cond)
        v_field = self.cinn(z, c=[self.flat_layer(cond)], rev=True)
        logj_rev = self.cinn.log_jacobian(run_forward=False)
        source = cond[:, :1, ...].to(self.device)
        target = cond[:, 1:, ...].to(self.device)

        deformation = self.integrator(v_field)
        target_pred = self.transformer(source, deformation)

        return target_pred, v_field, logj_rev

    def init_weights(self, method="gaussian"):
        """ This function does control the behaviour of the default weight initialization, by a set of keywords,
        currently being:
            - "gaussian" - for gaussian weight init
            - "xavier" - for weight init with the xavier-normal strategy
            - "uniform" - for uniform weights with the default pytorch mechanism. This is just default behaviour its
                          just an explicit tag for "stick to the default".
                          NOTE: Don't use this, for some reason this leads to incredibly big starting values for the

        """

        if method == "gaussian":
            def _init_gaussian(m):
                if isinstance(m, nn.Linear):
                    # torch.nn.init.normal_(m.weight, mean=0, std=1)
                    torch.nn.init.normal_(m.weight, std=0.01)
                    torch.nn.init.normal_(m.bias, std=0.01)

            self.apply(_init_gaussian)

            # for p in self.cinn.parameters():
            #     if p.requires_grad:
            #        p.data = 0.01 * torch.randn_like(p)

        elif method == "xavier":
            def _init_xavier(m):
                if isinstance(m, nn.Linear):
                    torch.nn.init.xavier_normal_(m.weight.data, gain=0.01)
                    torch.nn.init.constant_(m.bias.data, 0.0)
            self.apply(_init_xavier)
            last = self.cinn[-2:]

        elif method == "uniform":
            pass
            # layers ar initialized by default from a uniform distribution, see:
            # https://pytorch.org/docs/stable/generated/torch.nn.Linear.html#torch.nn.Linear

    def forward(self, x, c, rev=False):
        if self.cond_net:
            c = self.cond_net(c)
        z = self.cinn(x, [self.flat_layer(c)], rev=rev)
        log_j = self.cinn.log_jacobian(run_forward=False)
        return z, log_j

