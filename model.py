import torch
from torch import nn
import torch.nn.functional as F
from config import cfg



def conv(in_channel, out_channel, flag=None):
    if cfg.MODEL.ENCODER == 'CRNN' or cfg.MODEL.ENCODER == 'SymmetricalDualCRNN':
        # 单流或对称双流网络均使用3x3卷积
        return nn.Conv2d(in_channel, out_channel, kernel_size=3, padding=(1, 1))
    elif cfg.MODEL.ENCODER == 'AsymmetricDualCRNN':  # 非对称卷积
        if flag == 'x':
            return nn.Conv2d(in_channel, out_channel, kernel_size=(1, 3), padding=(0, 1))
        elif flag == 'y':
            return nn.Conv2d(in_channel, out_channel, kernel_size=(3, 1), padding=(1, 0))
        else:
            raise NotImplementedError
    else:
        raise NotImplementedError


def maxpool(flag=None):
    if flag == 'x':
        return nn.MaxPool2d((2, 1), (2, 1), padding=0)
    elif flag == 'y':
        return nn.MaxPool2d((1, 2), (1, 2), padding=0)
    else:
        raise NotImplementedError


class Net(nn.Module):
    def __init__(self, ):
        super(Net, self).__init__()
        if cfg.MODEL.ENCODER == 'DualAsymmetricCRNN':
            conv1_flag = 'x'
            conv2_flag = 'y'
        else:
            conv1_flag = None
            conv2_flag = None
        self.cnn1 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=(1, 1)), nn.ReLU(), nn.MaxPool2d(2, 2),
            conv(64, 128, 'x') if cfg.MODEL.FORCE_SYM_1 is False else nn.Conv2d(64, 128, kernel_size=(3, 3),
                                                                                padding=(1, 1)), nn.ReLU(),
            maxpool('x'),
            conv(128, 256, 'x') if cfg.MODEL.FORCE_SYM_2 is False else nn.Conv2d(128, 256, kernel_size=(3, 3),
                                                                                 padding=(1, 1)),
            nn.BatchNorm2d(256), nn.ReLU(),
            conv(256, 256, 'x') if cfg.MODEL.FORCE_SYM_3 is False else nn.Conv2d(256, 256, kernel_size=(3, 3),
                                                                                 padding=(1, 1)), nn.ReLU(),
            maxpool('x'),
            conv(256, 256, 'x'), nn.ReLU(), maxpool('x'),
            conv(256, 256, 'x'), nn.BatchNorm2d(256), nn.ReLU(),
            conv(256, 256, 'x'), nn.ReLU(), maxpool('x'),
            nn.Conv2d(256, 256, kernel_size=2, padding=(0, 1)), nn.BatchNorm2d(256), nn.ReLU()
            )
        is_bidirection = cfg.MODEL.BIDIRECTION
        self.rnn_encoder1 = nn.LSTM(256, 128 if is_bidirection else 256, bidirectional=is_bidirection, num_layers=3, batch_first=True)  # NTC TODO: bidirection
        self.rnn_decoder = nn.LSTM(256 + 5 if not cfg.MODEL.DECODE_WITHOUT_Z else 5, 256, num_layers=3,
                                   batch_first=True)  # NTC
        if cfg.MODEL.ENCODER == 'SymmetricalDualCRNN' or cfg.MODEL.ENCODER == 'AsymmetricDualCRNN':
            self.cnn2 = nn.Sequential(
                nn.Conv2d(1, 64, kernel_size=3, padding=(1, 1)), nn.ReLU(), nn.MaxPool2d(2, 2),
                conv(64, 128, 'y') if cfg.MODEL.FORCE_SYM_1 is False else nn.Conv2d(64, 128, kernel_size=(3, 3),
                                                                                    padding=(1, 1)), nn.ReLU(),
                maxpool('y'),
                conv(128, 256, 'y') if cfg.MODEL.FORCE_SYM_2 is False else nn.Conv2d(128, 256, kernel_size=(3, 3),
                                                                                     padding=(1, 1)),
                nn.BatchNorm2d(256), nn.ReLU(),
                conv(256, 256, 'y') if cfg.MODEL.FORCE_SYM_3 is False else nn.Conv2d(256, 256, kernel_size=(3, 3),
                                                                                     padding=(1, 1)), nn.ReLU(),
                maxpool('y'),
                conv(256, 256, 'y'), nn.ReLU(), maxpool('y'),
                conv(256, 256, 'y'), nn.BatchNorm2d(256), nn.ReLU(),
                conv(256, 256, 'y'), nn.ReLU(), maxpool('y'),
                nn.Conv2d(256, 256, kernel_size=2, padding=(1, 0)), nn.BatchNorm2d(256), nn.ReLU()
                )
            self.fusion = nn.Linear(512, 256)  # NTC
            self.rnn_encoder2 = nn.LSTM(256, 128 if is_bidirection else 256, bidirectional=is_bidirection, num_layers=3, batch_first=True)  # NTC

        else:
            self.cnn2 = None
            self.rnn_encoder2 = None
            self.fusion = None
        if cfg.MODEL.AttentionEN:
            self.attention = nn.Linear(256, 1)  # NTC
        else:
            self.attention = None
        self.atten_map = None

        self.output_layer = nn.Linear(256, 5)

    def decoder_cell(self, pre_point, pre_abs_xy, z, pre_hn, pre_cn, sk_points=None):#ps：decoder_cell只在测试阶段起作用
        cur_input = torch.cat((pre_point, z), dim=-1) if not cfg.MODEL.DECODE_WITHOUT_Z else pre_point
        raw_seq_pred, (x_hn, x_cn) = self.rnn_decoder(cur_input, (pre_hn, pre_cn)) #torch.Size([1, 1, 256])
        seq_pred = self.output_layer(raw_seq_pred) #torch.Size([1, 1, 5])
        if sk_points is not None:
            bs = len(sk_points)
            abs_xy_pred = pre_abs_xy + seq_pred[0, 0, [0, 1]]
            for b in range(bs):
                nearst_id = torch.square(abs_xy_pred[b].unsqueeze(0) - sk_points[b]).sum(-1).argmin()
                abs_xy_pred[b] = sk_points[b][nearst_id]
                seq_pred[b, 0, [0, 1]] = abs_xy_pred[b] - pre_abs_xy[b]
        else:
            # 对dx, dy取整
            seq_pred[:, :, [0, 1]] = torch.round(seq_pred[:, :, [0, 1]])
            abs_xy_pred = pre_abs_xy + seq_pred[:, 0, [0, 1]] #torch.Size([1, 2])
        # pred pen state [p1,p2,p3]
        seq_pred[:, :, [2, 3, 4]] = F.softmax(seq_pred[:, :, [2, 3, 4]], dim=-1)
        pen_state = torch.argmax(seq_pred[:, :, [2, 3, 4]], dim=-1) #torch.Size([1, 1])
        pen_state = torch.nn.functional.one_hot(pen_state, num_classes=3).to(seq_pred) #torch.Size([1, 1, 3])
        seq_pred[:, :, [2, 3, 4]] = pen_state
        return seq_pred, abs_xy_pred, x_hn, x_cn, raw_seq_pred

    def forward(self, x, seq=None, sk_points=None, max_step=100):
        # 编码
        x1 = self.cnn1(x)#torch.Size([32, 256, 1, 33])
        n, c, h, w = x1.size()
        x1 = x1.reshape(n, c, h * w).transpose(1, 2)#torch.Size([32, 33, 256])
        x1, (x_hn1, x_cn1) = self.rnn_encoder1(x1)
        if cfg.MODEL.BIDIRECTION:
            x_hn1, x_cn1 = x_hn1.view(3, 2, n, 128), x_cn1.view(3, 2, n, 128)
            x_hn1 = torch.cat([x_hn1[:, 0], x_hn1[:, -1]], dim=-1)
            x_cn1 = torch.cat([x_cn1[:, 0], x_cn1[:, -1]], dim=-1)
        if cfg.MODEL.ENCODER == 'SymmetricalDualCRNN' or cfg.MODEL.ENCODER == 'AsymmetricDualCRNN':
            # 双流网络
            x2 = self.cnn2(x)#torch.Size([32, 256, 33, 1])
            x2 = x2.transpose(2, 3).reshape(n, c, h * w).transpose(1, 2)#torch.Size([32, 33, 256])
            x2, (x_hn2, x_cn2) = self.rnn_encoder2(x2)
            if cfg.MODEL.BIDIRECTION:
                x_hn2, x_cn2 = x_hn2.view(3, 2, n, 128), x_cn2.view(3, 2, n, 128)
                x_hn2 = torch.cat([x_hn2[:, 0], x_hn2[:, -1]], dim=-1)
                x_cn2 = torch.cat([x_cn2[:, 0], x_cn2[:, -1]], dim=-1)
            x_hn = self.fusion(torch.cat((x_hn1, x_hn2), dim=-1))  # 对两个encoder输出进行降维，用于初始化decoder
            x_cn = self.fusion(torch.cat((x_cn1, x_cn2), dim=-1))
            x = torch.cat((x1, x2), dim=1)
            if not cfg.MODEL.AttentionEN:  # attention mechanmisim
                z = torch.sum(x, dim=1, keepdim=True)
            else:
                weight = F.softmax(self.attention(x), dim=1)
                self.atten_map = weight if not self.training else None
                x, weight = torch.broadcast_tensors(x, weight)
                x = x * weight
                z = torch.sum(x, dim=1, keepdim=True)  
        else:
            # 单流网络
            if cfg.MODEL.AttentionEN:
                x = x1
                weight = F.softmax(self.attention(x), dim=1)
                self.atten_map = weight if not self.training else None
                x, weight = torch.broadcast_tensors(x, weight)
                x1 = x * weight
            z = torch.sum(x1, dim=1, keepdim=True)#32,1,256
            x_hn, x_cn = x_hn1, x_cn1 #3,32,256

        #  解码
        if self.training:
            n, t, c = seq.size()    #32,110,5
            z = torch.repeat_interleave(z, repeats=t, dim=1) ##32,110,256
            if not cfg.MODEL.DECODE_WITHOUT_Z:
                seq = torch.cat((seq, z), dim=2)
            seq_pred, _ = self.rnn_decoder(seq, (x_hn, x_cn))
            seq_pred = self.output_layer(seq_pred) #torch.Size([32, 117, 256])
            return seq_pred #torch.Size([32, 117, 5])
        else:
            # assert x.size(0) == 1, 'only support bs=1 inference'
            all_preds = torch.zeros((n, max_step, 5)).to(x) #torch.Size([1, 36, 5])
            sos = torch.Tensor(n * [[[0, 0, 1, 0, 0]]]).to(x)
            abs_xy_init = torch.zeros((n, 2)).to(x)
            dxdy_pred, abs_xy_pred, x_hn, x_cn, raw_seq = self.decoder_cell(pre_point=sos, pre_abs_xy=abs_xy_init, z=z,
                                                                            pre_hn=x_hn, pre_cn=x_cn,
                                                                            sk_points=sk_points)
            all_preds[:, 0] = dxdy_pred[:, 0]
            for i in range(1, max_step):
                pre_point = all_preds[:, i - 1].unsqueeze(1)
                dxdy_pred, abs_xy_pred, x_hn, x_cn, raw_seq = self.decoder_cell(pre_point=pre_point, #torch.Size([1, 1, 5]) #torch.Size([1, 2])
                                                                                pre_abs_xy=abs_xy_pred,
                                                                                z=z, pre_hn=x_hn, pre_cn=x_cn,
                                                                                sk_points=sk_points)
                all_preds[:, i] = dxdy_pred[:, 0]
                if dxdy_pred[:, 0, -1].sum() == n:
                    # 所有样本都已译码结束，提前终止
                    break
            return all_preds
if __name__ == '__main__':
    """单元测试：
    1. CRNN w/o attention
    2. SymmetricalDualCRNN with attention
    3. SymmetricalDualCRNN w/o attention
    4. AsymmetricDualCRNN with attention
    5. AsymmetricDualCRNN w/o attention
    """

    img = torch.zeros((32, 1, 64, 64))
    seq = torch.zeros((32, 100, 5))
    # test1
    test_params = [('CRNN', False),
                   ('SymmetricalDualCRNN', True),
                   ('SymmetricalDualCRNN', False),
                   ('AsymmetricDualCRNN', True),
                   ('AsymmetricDualCRNN', False)
                   ]
    for i, (p1, p2) in enumerate(test_params):
        cfg.MODEL.ENCODER = p1
        cfg.MODEL.AttentionEN = p2
        model = Net()
        model(img, seq)
        # summary(model, img, seq)
        print('test {} passed\n'.format(i + 1))
