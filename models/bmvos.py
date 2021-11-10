from collections import OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision as tv


def softmax_aggregate(predicted_seg, object_ids):
    bg_seg, _ = torch.stack([seg[:, 0, :, :] for seg in predicted_seg.values()], dim=1).min(dim=1)
    bg_seg = torch.stack([1 - bg_seg, bg_seg], dim=1)
    logits = {n: seg[:, 1:, :, :].clamp(1e-7, 1 - 1e-7) / seg[:, 0, :, :].clamp(1e-7, 1 - 1e-7)
              for n, seg in [(-1, bg_seg)] + list(predicted_seg.items())}
    logits_sum = torch.cat(list(logits.values()), dim=1).sum(dim=1, keepdim=True)
    aggregated_lst = [logits[n] / logits_sum for n in [-1] + object_ids]
    aggregated_inv_lst = [1 - elem for elem in aggregated_lst]
    aggregated = torch.cat([elem for lst in zip(aggregated_inv_lst, aggregated_lst) for elem in lst], dim=-3)
    final_seg_wrongids = aggregated[:, 1::2, :, :].argmax(dim=-3, keepdim=True)
    assert final_seg_wrongids.dtype == torch.int64
    final_seg = torch.zeros_like(final_seg_wrongids)
    for idx, obj_idx in enumerate(object_ids):
        final_seg[final_seg_wrongids == (idx + 1)] = obj_idx
    return final_seg, {obj_idx: aggregated[:, 2 * (idx + 1):2 * (idx + 2), :, :] for idx, obj_idx in enumerate(object_ids)}


def get_required_padding(height, width, div):
    height_pad = (div - height % div) % div
    width_pad = (div - width % div) % div
    padding = [(width_pad + 1) // 2, width_pad // 2, (height_pad + 1) // 2, height_pad // 2]
    return padding


def apply_padding(x, y, padding):
    B, L, C, H, W = x.size()
    x = x.view(B * L, C, H, W)
    x = F.pad(x, padding, mode='reflect')
    _, _, height, width = x.size()
    x = x.view(B, L, C, height, width)
    y = [F.pad(label.float(), padding, mode='reflect').long() if label is not None else None for label in y]
    return x, y


def unpad(tensor, padding):
    if isinstance(tensor, (dict, OrderedDict)):
        return {key: unpad(val, padding) for key, val in tensor.items()}
    elif isinstance(tensor, (list, tuple)):
        return [unpad(elem, padding) for elem in tensor]
    else:
        _, _, _, height, width = tensor.size()
        tensor = tensor[:, :, :, padding[2]:height - padding[3], padding[0]:width - padding[1]]
        return tensor


class Conv(nn.Sequential):
    def __init__(self, *conv_args):
        super().__init__()
        self.add_module('conv', nn.Conv2d(*conv_args))
        for m in self.children():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)


class ConvRelu(nn.Sequential):
    def __init__(self, *conv_args):
        super().__init__()
        self.add_module('conv', nn.Conv2d(*conv_args))
        self.add_module('naf', nn.ReLU(inplace=True))
        for m in self.children():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)


class DeConv(nn.Sequential):
    def __init__(self, *conv_args):
        super().__init__()
        self.add_module('deconv', nn.ConvTranspose2d(*conv_args))
        for m in self.children():
            if isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)


class AddCoords(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input_tensor):
        batch_size_tensor = input_tensor.size(0)
        x_dim = input_tensor.size(3)
        y_dim = input_tensor.size(2)

        xx_ones = torch.ones([1, x_dim], dtype=torch.int32)
        xx_range = torch.arange(y_dim, dtype=torch.int32).unsqueeze(1)
        xx_channel = torch.matmul(xx_range, xx_ones).unsqueeze(0)

        yy_ones = torch.ones([y_dim, 1], dtype=torch.int32)
        yy_range = torch.arange(x_dim, dtype=torch.int32).unsqueeze(0)
        yy_channel = torch.matmul(yy_ones, yy_range).unsqueeze(0)

        xx_channel = xx_channel.float() / (y_dim - 1)
        yy_channel = yy_channel.float() / (x_dim - 1)

        xx_channel = xx_channel * 2 - 1
        yy_channel = yy_channel * 2 - 1

        xx_channel = xx_channel.repeat(batch_size_tensor, 1, 1, 1)
        yy_channel = yy_channel.repeat(batch_size_tensor, 1, 1, 1)

        xx_channel = xx_channel.cuda()
        yy_channel = yy_channel.cuda()

        ret = torch.cat([input_tensor, xx_channel, yy_channel], dim=1)
        rr = torch.sqrt(torch.pow(xx_channel, 2) + torch.pow(yy_channel, 2))
        ret = torch.cat([ret, rr], dim=1)
        return ret


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        backbone = tv.models.densenet121(pretrained=True).features
        self.conv0 = backbone.conv0
        self.norm0 = backbone.norm0
        self.relu0 = backbone.relu0
        self.pool0 = backbone.pool0
        self.denseblock1 = backbone.denseblock1
        self.transition1 = backbone.transition1
        self.denseblock2 = backbone.denseblock2
        self.transition2 = backbone.transition2
        self.denseblock3 = backbone.denseblock3
        self.register_buffer('mean', torch.FloatTensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.FloatTensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def get_features(self, x):
        x = (x - self.mean) / self.std
        x = self.conv0(x)
        x = self.norm0(x)
        x = self.relu0(x)
        x = self.pool0(x)
        x = self.denseblock1(x)
        s4 = x
        x = self.transition1(x)
        x = self.denseblock2(x)
        s8 = x
        x = self.transition2(x)
        x = self.denseblock3(x)
        s16 = x
        return {'s16': s16, 's8': s8, 's4': s4}


class Matcher(nn.Module):
    def __init__(self, indim, outdim):
        super().__init__()
        self.conv = Conv(indim, outdim, 1, 1, 0)

    def get_conv_feats_norm(self, feats):
        conv_feats = self.conv(feats)
        conv_feats_norm = conv_feats / conv_feats.norm(dim=1, keepdim=True)
        return conv_feats_norm

    def forward(self, init_sim, prev_sim, state):
        B, _, H, W = init_sim.size()

        bg_sim = init_sim * state['init_seg'][:, 0].view(B, H * W, 1, 1)
        bg_sim, _ = torch.max(bg_sim, dim=1, keepdim=True)
        fg_sim = init_sim * state['init_seg'][:, 1].view(B, H * W, 1, 1)
        fg_sim, _ = torch.max(fg_sim, dim=1, keepdim=True)
        global_sim = torch.cat([bg_sim, fg_sim], dim=1)

        K = 4

        score = prev_sim * state['prev_seg'][:, 0].view(B, H * W, 1, 1)
        score = score.view(B, H * W, H * W)
        topk, _ = torch.topk(score, k=K, dim=-1, sorted=True)
        cut = topk[:, :, -1:]
        min, _ = torch.min(score, dim=-1, keepdim=True)
        cut = cut.repeat(1, 1, H * W)
        min = min.repeat(1, 1, H * W)
        score[score < cut] = 0
        score = (score - min).clamp(min=0) + min
        score = score.view(B, H * W, H, W)
        bg_sim, _ = torch.max(score, dim=1, keepdim=True)

        score = prev_sim * state['prev_seg'][:, 1].view(B, H * W, 1, 1)
        score = score.view(B, H * W, H * W)
        topk, _ = torch.topk(score, k=K, dim=-1, sorted=True)
        cut = topk[:, :, -1:]
        min, _ = torch.min(score, dim=-1, keepdim=True)
        cut = cut.repeat(1, 1, H * W)
        min = min.repeat(1, 1, H * W)
        score[score < cut] = 0
        score = (score - min).clamp(min=0) + min
        score = score.view(B, H * W, H, W)
        fg_sim, _ = torch.max(score, dim=1, keepdim=True)
        local_sim = torch.cat([bg_sim, fg_sim], dim=1)

        total_sim = torch.cat([global_sim, local_sim], dim=1)
        return total_sim


class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = ConvRelu(1024, 256, 1, 1, 0)
        self.blend1 = ConvRelu(256 + 4 + 256, 256, 3, 1, 1)
        self.deconv1 = DeConv(256, 2, 4, 2, 1)
        self.conv2 = ConvRelu(512, 256, 1, 1, 0)
        self.blend2 = ConvRelu(256 + 2, 256, 3, 1, 1)
        self.deconv2 = DeConv(256, 2, 4, 2, 1)
        self.conv3 = ConvRelu(256, 256, 1, 1, 0)
        self.blend3 = ConvRelu(256 + 2, 256, 3, 1, 1)
        self.predictor = DeConv(256, 2, 6, 4, 1)

    def forward(self, feats, sim, mask_feats):
        u = torch.cat([self.conv1(feats['s16']), sim, mask_feats], dim=1)
        u = self.blend1(u)
        out_8 = self.deconv1(u)
        u = torch.cat([self.conv2(feats['s8']), out_8], dim=1)
        u = self.blend2(u)
        out_4 = self.deconv2(u)
        u = torch.cat([self.conv3(feats['s4']), out_4], dim=1)
        u = self.blend3(u)
        segscore = self.predictor(u)
        return segscore


class VOS(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = Encoder()
        self.matcher = Matcher(1024, 512)
        self.decoder = Decoder()
        self.coorder = AddCoords()
        self.mask_encoder = nn.Sequential(
            ConvRelu(9, 32, 7, 2, 3), ConvRelu(32, 64, 7, 2, 3),
            ConvRelu(64, 128, 7, 2, 3), Conv(128, 256, 7, 2, 3))

    def get_init_state(self, given_seg, given_seg_16, conv_feats_norm):
        state = {}
        state['masks'] = [given_seg, given_seg, given_seg]
        state['mask_feats'] = self.extract_mask_feats(state['masks'])
        state['init_seg'] = given_seg_16
        state['prev_seg'] = given_seg_16
        state['init_conv_feats_norm'] = conv_feats_norm
        state['prev_conv_feats_norm'] = conv_feats_norm
        return state

    def update(self, update_seg, update_seg_16, conv_feats_norm, full_state, object_ids):
        for id in object_ids:
            full_state[id]['masks'].append(update_seg[id])
            del full_state[id]['masks'][0]
            full_state[id]['mask_feats'] = self.extract_mask_feats(full_state[id]['masks'])
            full_state[id]['prev_seg'] = update_seg_16[id]
            full_state[id]['prev_conv_feats_norm'] = conv_feats_norm
        return full_state

    def extract_feats(self, img):
        feats = self.encoder.get_features(img)
        return feats

    def extract_mask_feats(self, masks):
        x = torch.cat([masks[-1], masks[-2], masks[-3]], dim=1)
        x = self.coorder(x)
        x = self.mask_encoder(x)
        return x

    def forward(self, conv_feats_norm, feats, full_state, object_ids):
        B, _, H, W = conv_feats_norm.size()
        segscore = {}
        for id in object_ids:
            init_conv_feats_norm = full_state[id]['init_conv_feats_norm'].view(B, -1, H * W).transpose(1, 2)
            prev_conv_feats_norm = full_state[id]['prev_conv_feats_norm'].view(B, -1, H * W).transpose(1, 2)

            init_sim = (torch.bmm(init_conv_feats_norm, conv_feats_norm.view(B, -1, H * W)).view(B, H * W, H, W) + 1) / 2
            prev_sim = (torch.bmm(prev_conv_feats_norm, conv_feats_norm.view(B, -1, H * W)).view(B, H * W, H, W) + 1) / 2

            sim = self.matcher(init_sim, prev_sim, full_state[id])
            segscore[id] = self.decoder(feats, sim, full_state[id]['mask_feats'])
        return segscore


class BMVOS(nn.Module):
    def __init__(self):
        super().__init__()
        self.vos = VOS()

    def forward(self, x, given_masks=None, val_frame_ids=None):
        B, L, _, H0, W0 = x.size()

        required_padding = get_required_padding(H0, W0, 16)
        if tuple(required_padding) != (0, 0, 0, 0):
            x, given_masks = apply_padding(x, given_masks, required_padding)
        _, _, _, H1, W1 = x.size()
        video_frames = [elem.view(B, 3, H1, W1) for elem in x.split(1, dim=1)]
        feats = self.vos.extract_feats(video_frames[0])
        conv_feats_norm = self.vos.matcher.get_conv_feats_norm(feats['s16'])
        init_mask = given_masks[0]
        object_ids = init_mask.unique().tolist()
        if 0 in object_ids:
            object_ids.remove(0)

        state = {}
        for k in object_ids:
            given_seg = torch.cat([init_mask != k, init_mask == k], dim=1).float()
            given_seg_16 = F.avg_pool2d(given_seg, 16)
            state[k] = self.vos.get_init_state(given_seg, given_seg_16, conv_feats_norm)
        seg_lst = []

        seg_lst.append(given_masks[0])
        frames_to_process = range(1, L)

        for i in frames_to_process:

            feats = self.vos.extract_feats(video_frames[i])
            conv_feats_norm = self.vos.matcher.get_conv_feats_norm(feats['s16'])

            segscore = self.vos(conv_feats_norm, feats, state, object_ids)
            predicted_seg = {k: F.softmax(segscore[k], dim=1) for k in object_ids}

            if given_masks[i] is not None:
                new_object_ids = given_masks[i].unique().tolist()
                if 0 in new_object_ids:
                    new_object_ids.remove(0)
                for new_k in new_object_ids:
                    given_seg = torch.cat([given_masks[i] != new_k, given_masks[i] == new_k], dim=1).float()
                    given_seg_16 = F.avg_pool2d(given_seg, 16)
                    state[new_k] = self.vos.get_init_state(given_seg, given_seg_16, conv_feats_norm)
                    predicted_seg[new_k] = torch.cat([given_masks[i] != new_k, given_masks[i] == new_k], dim=1).float()
                object_ids = object_ids + new_object_ids

            output_seg, aggregated_seg = softmax_aggregate(predicted_seg, object_ids)
            update_seg = aggregated_seg
            update_seg_16 = {k: F.avg_pool2d(aggregated_seg[k], 16) for k in object_ids}

            if i < L-1:
                state = self.vos.update(update_seg, update_seg_16, conv_feats_norm, state, object_ids)

            if isinstance(given_masks, (list, tuple)) and given_masks[i] is not None:
                output_seg[given_masks[i] != 0] = 0
                seg_lst.append(output_seg + given_masks[i])
            else:
                if val_frame_ids is not None:
                    adjusted_i = i + val_frame_ids[0]
                    if adjusted_i in val_frame_ids:
                        seg_lst.append(output_seg)
                else:
                    seg_lst.append(output_seg)

        output = {}
        output['segs'] = torch.stack(seg_lst, dim=1)
        output['segs'] = unpad(output['segs'], required_padding)
        return output

