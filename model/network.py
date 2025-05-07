"""
视觉提示调优VPT---第一版--pass

"""


import os

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
from argparse import Namespace
import timm
import torch
import torch.nn.functional as F
from torch import nn
from model.model import build_model as CLIPModel

# 构建模型函数
def build_model(args, pretrained=True):
    if args.backbone == "vit":
        net = Network(args, pretrained)
    else:
        raise NotImplementedError(f"not support: {args.backbone}")
    return net.cuda(), 0


class LinearHash(nn.Module):

    def __init__(self, inputDim=2048, outputDim=64):
        super(LinearHash, self).__init__()
        self.fc = nn.Linear(inputDim, outputDim)
        # self.fc.apply(weights_init_kaiming)
        self.drop_out = nn.Dropout(p=0.2)

    def forward(self, data):
        result = self.fc(data)
        return torch.tanh(self.drop_out(result))

# 定义主网络
class Network(nn.Module):
    """
    Notations:
    B: batch_size
    C: n_classes
    K: n_bits
    D: self.model.embed_dim
    N: n_prompts or n_p_prompts
    """

    def __init__(self, args: Namespace, pretrained):
        super(Network, self).__init__()
        self.args = args
        # 使用timm库创建ViT模型
        self.model = timm.create_model("vit_small_patch16_224", pretrained=pretrained)

        # freezing 冻结模型参数
        for param in self.model.parameters():
            param.requires_grad = False

        # setting bitfit---只训练偏置
        if args.vit_bitfit:
            for n, param in self.model.named_parameters():
                if self._check_params(n, ["bias"], all_match=False):
                    param.requires_grad = True

        # for features--特征提取层
        self.head = nn.Linear(self.model.embed_dim, args.n_bits).apply(self._init_weights)
        # note: self.model.fc_norm = Identity()
        self.fc_norm = nn.LayerNorm(self.model.embed_dim).apply(self._init_weights)
        self.prompts = self._setup_prompt(args.prompt_depth, 1, args.n_prompts)

        # for proxies--代理生成层
        self.p_head = nn.Linear(self.model.embed_dim, args.n_bits).apply(self._init_weights)
        self.p_fc_norm = nn.LayerNorm(self.model.embed_dim).apply(self._init_weights)
        self.p_prompts = self._setup_prompt(args.p_prompt_depth, args.n_classes, args.n_p_prompts)

        # proxies container & bias (O in paper) -> used as centroids container-----代理容器和偏置
        self.proxies_container = nn.Parameter(
            F.normalize(nn.init.xavier_uniform_(torch.zeros(args.n_samples, args.n_bits)) / 8.0, dim=1),
            requires_grad=False,
        )
        self.proxies_bias = nn.Parameter(nn.init.xavier_uniform_(torch.zeros(args.n_samples, args.n_bits)) / 8.0)

        embedDim, self.clip = self.load_clip(args.clipPath)

        self.text_hash = LinearHash(inputDim=embedDim, outputDim=args.n_bits)

        # 简单GRU单元，用于代理更新
        if args.semantic_mix_type is not None:
            self.rnn_update = Simple_GRUCell(args.n_bits, args.n_bits, drop=args.rnn_dropout, type=args.semantic_mix_type)

    def load_clip(self, clipPath: str) -> tuple:
        try:
            model = torch.jit.load(clipPath, map_location="cpu").eval()
            state_dict = model.state_dict()
        except RuntimeError:
            state_dict = torch.load(clipPath, map_location="cpu")
        embed_dim = state_dict["text_projection"].shape[1]
        clip_model = CLIPModel(state_dict)

        return embed_dim, clip_model

    # check params name---检查参数是否在允许的列表中
    def _check_params(self, module_name, safe_list, all_match=True):
        check = [partial_name in module_name for partial_name in safe_list]
        return all(check) if all_match else any(check)

    # 初始化权重
    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.LayerNorm):
            nn.init.constant_(module.weight, 1.0)
            nn.init.constant_(module.bias, 0)
        else:
            raise NotImplementedError("not init implemented for {}".format(module))

    # 设置提示参数
    def _setup_prompt(self, depth, length, n_prompts):
        """
        setup VPTs
        """
        # depth x length x N x D
        prompts = nn.ParameterList(
            [
                nn.Parameter(nn.init.xavier_normal_(torch.zeros(length, n_prompts, self.model.embed_dim)))  # 新增加  , dtype=torch.float32
                for _ in range(depth)
            ]
        )
        return prompts

    # 处理VPT块
    def _forward_vpt_block(self, x, unit, vpt):
        s = x.shape[1]
        x = torch.cat([vpt, x], dim=1)
        x = unit(x)
        x = x[:, vpt.shape[1] :, :]
        assert x.shape[1] == s
        return x

    # 前向传播特征提取
    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        # print(x.shape)
        x = self.model.patch_embed(x)  # B x D x 14 x 14
        # print(x.shape)
        x = self.model._pos_embed(x)  # B x 197 x D
        # print(x.shape)
        # x = self.model.patch_drop(x)  # Identity()
        x = self.model.norm_pre(x)  # B x 197 x D
        # print(x.shape)
        # x = self.blocks(x) -> Visual Prompt Tuning
        for idx, blk in enumerate(self.model.blocks):
            if idx < self.args.prompt_depth:
                prompt = self.prompts[idx]
                prompt = prompt.repeat(x.shape[0], 1, 1)  # 1 x N x D -> B x N x D
                x = self._forward_vpt_block(x, blk, prompt)
            else:
                x = blk(x)
        x = self.model.norm(x)

        return x

    # 前向传播头部处理
    def forward_head(self, x: torch.Tensor) -> torch.Tensor:
        x = x[:, 0]  # cls token
        x = self.fc_norm(x)
        x = self.head(x)
        x = F.normalize(x, dim=-1)  # diff from origin ViT
        return x

    # 使用容器进行语义混合
    def _semantic_mix_with_container(self, p, index):
        """
        fuse the proxies with container of history proxies
        """
        # C x D, B x C -> B x D
        print(self.proxies_container[index].shape)
        print(self.proxies_container[index])
        exit()
        p_prev = self.proxies_container[index]  #
        # print(p.shape)
        # print(p_prev.shape)
        # exit()
        p = self.rnn_update(p, p_prev)
        self.proxies_container[index] = p.detach()
        self.proxies_container = nn.Parameter(F.normalize(self.proxies_container, dim=-1), requires_grad=False)
        return p

    # 生成代理
    def _proxies_generator(self, p, index):
        p = p[:, 0]  # B x D
        p = self.p_fc_norm(p)
        p = self.p_head(p)  # B x D -> B x K
        p = F.normalize(p, dim=-1)
        # GRU
        if self.args.semantic_mix_type is not None:
            p = self._semantic_mix_with_container(p, index)
        alpha = self.args.proxy_bias_ratio
        # B x C, C x K -> B x K
        p = (1 - alpha) * p + alpha * (self.proxies_bias[index])
        p = F.normalize(p, dim=-1)
        return p

    # 前向传播生成代理，包含偏置
    def forward_proxy_mix_bias(self, images, labels, index):
        # pre_embedding
        p = self.model.patch_embed(images)  # B x D x 14 x 14
        p = self.model._pos_embed(p)  # B x 197 x D
        # p = self.model.patch_drop(p)  # Identity()
        p = self.model.norm_pre(p)  # B x 197 x D
        # p = self.blocks(p) -> proxy net
        batch_p_prompts = []

        for idx in range(self.args.p_prompt_depth):
            # B x C, C x N x D -> B x N x D
            # print(labels.shape)
            # print(self.p_prompts[idx].shape)
            # exit()

            self.p_prompts[idx] = self.p_prompts[idx].float()
            labels = labels.float()
            # print(labels.dtype)
            # print(self.p_prompts[idx].dtype)
            # print(self.p_prompts[idx])
            # exit()

            p_prompt = torch.einsum("bc,cnd->bnd", (labels, self.p_prompts[idx]))
            p_prompt /= labels.sum(1, keepdim=True).unsqueeze(-1)  # calc avg value
            batch_p_prompts.append(p_prompt)

        for idx, blk in enumerate(self.model.blocks):
            # transfer i to block index
            # add vpt with the running blocks
            if idx < self.args.p_prompt_depth:
                p_prompt = batch_p_prompts[idx]  # B x N x D
                p = self._forward_vpt_block(p, blk, p_prompt)
            else:
                p = blk(p)
        p = self.model.norm(p)  # B x 197 x D
        return self._proxies_generator(p, index)
    def encode_text(self, text):

        text_embed = self.clip.encode_text(text)
        text_embed = text_embed.to(self.text_hash.fc.weight.dtype)
        text_embed = self.text_hash(text_embed)

        return text_embed

    # 前向传播主函数
    def forward(self, images, texts, labels=None, index=None):
        # print(images.shape)  # torch.Size([128, 3, 224, 224])
        images= images.float()

        x = self.forward_features(images)
        x = self.forward_head(x)

        t = self.encode_text(texts)

        # if self.training:
            # centroids
            # p = self.forward_proxy_mix_bias(images, labels, index)
        # else:
            # p = None

        return x, t


class Simple_GRUCell(nn.Module):
    def __init__(self, input_size, hidden_size, drop=0.0, type="gru"):
        super(Simple_GRUCell, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.rnn_drop = nn.Dropout(drop)
        self.type = type

        self.W_z = nn.Parameter(torch.Tensor(input_size, hidden_size))
        self.U_z = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        self.b_z = nn.Parameter(torch.Tensor(hidden_size))

        self.W_r = nn.Parameter(torch.Tensor(input_size, hidden_size))
        self.U_r = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        self.b_r = nn.Parameter(torch.Tensor(hidden_size))

        self.W_h = nn.Parameter(torch.Tensor(input_size, hidden_size))
        self.U_h = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        self.b_h = nn.Parameter(torch.Tensor(hidden_size))

        self.init_weights()

    def init_weights(self):
        for p in self.parameters():
            if p.data.ndimension() >= 2:
                nn.init.xavier_uniform_(p.data)
            else:
                nn.init.zeros_(p.data)

    def forward(self, p, p_prev):
        z = torch.sigmoid(p @ self.W_z + p_prev @ self.U_z + self.b_z)
        r = torch.sigmoid(p @ self.W_r + p_prev @ self.U_r + self.b_r)

        if self.type == "gru":
            n = torch.tanh(p @ self.W_h + r * (p_prev @ self.U_h) + self.b_h)
        elif self.type == "gru_relu":
            n = F.relu(p @ self.W_h + r * (p_prev @ self.U_h) + self.b_h)
        else:
            raise NotImplementedError
        # p_next = (1 - z) * n + z * p_prev # diff from paper
        p_next = (1 - z) * p_prev + z * n  # Eq. (7) without normal()
        p_next = self.rnn_drop(p_next)
        return p_next


if __name__ == "__main__":
    _batch_size = 8
    _args = Namespace(
        n_bits=16,
        n_classes=10,
        n_samples=100,
        prompt_depth=12,
        p_prompt_depth=12,  # 3->12
        n_prompts=10,
        n_p_prompts=3,  # 5->3
        semantic_mix_type="gru_relu",
        rnn_dropout=0.0,
        proxy_bias_ratio=0.5,
        vit_bitfit=True,
    )
    net = Network(_args, True).cuda()

    # for param in net.named_parameters():
    #     print(param[0], param[1].requires_grad)

    other_params = [
        x[-1] for x in list(filter(lambda x: "proxies_bias" not in x[0] and x[1].requires_grad, net.named_parameters()))
    ]
    print(len(other_params))
    # print(net.proxies_bias)

    # _images = torch.randn((_batch_size, 3, 224, 224)).cuda()
    # _targets = torch.randint(_args.n_classes, (_batch_size,)).cuda()
    # _labels = F.one_hot(_targets, _args.n_classes).float()

    # logits, proxies = net(_images, _labels, torch.randint(_args.n_samples, (_batch_size,)))
    # print(logits.shape)
    # print(proxies.shape)
    # print(net.model.fc_norm)
    # print(net.fc_norm)

    # print(net)
